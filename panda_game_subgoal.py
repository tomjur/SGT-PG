import os
from random import Random
import numpy as np
import multiprocessing
import queue
import time

from panda_scene_manager import PandaSceneManager
from abstract_motion_planning_game_subgoal import AbstractMotionPlanningGameSubgoal


class PandaGameSubgoal(AbstractMotionPlanningGameSubgoal):
    def __init__(self, scenario, max_cores=None):
        self.scenario = scenario
        self.panda_scene_manager = PandaSceneManager.get_scene_manager(scenario)
        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()

        self.workers = [
            GameWorker(scenario, self.requests_queue, self.results_queue)
            for _ in range(self._get_number_of_workers(max_cores))
        ]

        for w in self.workers:
            w.start()

    @staticmethod
    def _get_number_of_workers(max_cores):
        # return 1  # for debug: set one worker
        # get available cpus
        cores = multiprocessing.cpu_count()
        max_cores_with_slack = max(cores - 2, 1)
        if max_cores is not None:
            # the number of workers is the min between those
            return min(max_cores, max_cores_with_slack)
        else:
            return max_cores_with_slack

    @staticmethod
    def _truncate_virtual_state(state):
        truncated_state = np.maximum(np.minimum(state, 1.), -1.)
        truncated_distance = np.linalg.norm(state - truncated_state)
        return truncated_state, truncated_distance

    def is_free_state(self, state):
        return self.is_free_state_in_manager(state, self.panda_scene_manager)

    @staticmethod
    def is_free_state_in_manager(state, panda_scene_manager):
        if any(np.abs(state) > 1.0):
            return False
        state_ = PandaGameSubgoal.virtual_to_real_state(state, panda_scene_manager)
        panda_scene_manager.change_robot_joints(state_)
        is_collision = panda_scene_manager.simulation_step()[1]
        if not panda_scene_manager.is_close(state_):
            return False
        if is_collision:
            return False
        return True

    def get_fixed_start_goal_pairs(self):
        # for now a long straight movement
        lower = np.array(self.panda_scene_manager.joints_lower_bounds).copy()
        upper = np.array(self.panda_scene_manager.joints_upper_bounds).copy()

        if self.scenario == 'panda_no_obs':
            s = self.real_to_virtual_state(0.5 * upper + 0.5 * lower, self.panda_scene_manager)
            g = self.real_to_virtual_state(upper.copy(), self.panda_scene_manager)
        elif self.scenario == 'panda_easy':
            s = upper.copy()
            s[1] += lower[1]
            s[1] *= 0.5
            s[3] += lower[3]
            s[3] *= 0.5
            s = self.real_to_virtual_state(s, self.panda_scene_manager)

            g = upper.copy()
            g[0] = 0.7 * upper[0] + 0.3 * lower[0]
            g[2] = 0.1 * upper[2] + 0.9 * lower[2]
            g[3] = 0.4 * upper[3] + 0.6 * lower[3]
            g[4] = 0.7 * upper[4] + 0.3 * lower[4]
            g = self.real_to_virtual_state(g, self.panda_scene_manager)
        elif self.scenario == 'panda_hard':
            s = upper.copy()
            s[0] = 0.7 * upper[0] + 0.3 * lower[0]
            s[1] = 0.7 * upper[1] + 0.3 * lower[1]
            s[2] = 0.1 * upper[2] + 0.9 * lower[2]
            s[3] = 0.4 * upper[3] + 0.6 * lower[3]
            s[4] = 0.7 * upper[4] + 0.3 * lower[4]
            s[5] = 0.7 * upper[5] + 0.3 * lower[5]
            s[6] = 0.7 * upper[6] + 0.3 * lower[6]
            s = self.real_to_virtual_state(s, self.panda_scene_manager)

            g = upper.copy()
            g[0] = 0.1 * upper[0] + 0.9 * lower[0]
            g[1] = 0.6 * upper[1] + 0.4 * lower[1]
            g[3] = 0.1 * upper[3] + 0.9 * lower[3]
            g[4] = 0.4 * upper[4] + 0.6 * lower[4]
            g = self.real_to_virtual_state(g, self.panda_scene_manager)
        else:
            assert False

        assert self.is_free_state(s)
        assert self.is_free_state(g)
        return [(s, g)]

    def test_predictions(self, cost_queries):
        # put all requests
        for path_id in cost_queries:
            path_requests = (path_id, 1, tuple(cost_queries[path_id]))
            self.requests_queue.put(path_requests)

        # pull all responses
        results = {}
        for _ in range(len(cost_queries)):
            response = self.results_queue.get(block=True)
            path_id, path_results = response
            results[path_id] = {t[0]: t[1] for t in path_results}
        return results

    def get_free_start_goals(self, number_of_episodes, curriculum_coefficient):
        # put all requests
        for i in range(number_of_episodes):
            self.requests_queue.put((i, 0, curriculum_coefficient))

        # pull all responses
        results = []
        for _ in range(number_of_episodes):
            response = self.results_queue.get(block=True)
            results.append(response)
        return results

    def get_state_size(self):
        return self.panda_scene_manager.number_of_joints

    @staticmethod
    def virtual_to_real_state(s, panda_scene_manager):
        lower = np.array(panda_scene_manager.joints_lower_bounds)
        upper = np.array(panda_scene_manager.joints_upper_bounds)
        s_ = np.array(s)
        s_ = (upper - lower) * s_ + (upper + lower)
        s_ = s_ / 2.
        return s_

    @staticmethod
    def real_to_virtual_state(s, panda_scene_manager):
        lower = np.array(panda_scene_manager.joints_lower_bounds)
        upper = np.array(panda_scene_manager.joints_upper_bounds)
        denom = upper - lower
        s_ = np.array(s)
        s_ = 2 * s_ - (upper + lower)
        s_ = s_ / denom
        return s_


class GameWorker(multiprocessing.Process):
    def __init__(self, scenario, requests_queue, results_queue):
        multiprocessing.Process.__init__(self)
        self.scenario = scenario
        self.requests_queue = requests_queue
        self.results_queue = results_queue

        self._panda_scene_manager = None
        self._random = None

    def run(self):
        self._random = Random(os.getpid())
        self._panda_scene_manager = PandaSceneManager.get_scene_manager(self.scenario)
        while True:
            try:
                path_id, message_type, params = self.requests_queue.get(block=True, timeout=0.001)
                if message_type == 0:
                    # get a valid random state
                    response = self.get_valid_start_goal(params)
                elif message_type == 1:
                    # check terminal segments
                    path_cost_results = [
                        (i, self.check_terminal_segment(start, end))
                         for i, start, end in params
                    ]
                    response = (path_id, path_cost_results)
                else:
                    assert False
                self.results_queue.put(response)
            except queue.Empty:
                time.sleep(1.0)

    def check_terminal_segment(self, start, end):
        self._panda_scene_manager.reset_simulation()
        # check the end points
        truncated_start, truncated_distance_start = PandaGameSubgoal._truncate_virtual_state(start)
        truncated_end, truncated_distance_end = PandaGameSubgoal._truncate_virtual_state(end)

        start_ = PandaGameSubgoal.virtual_to_real_state(truncated_start, self._panda_scene_manager)
        end_ = PandaGameSubgoal.virtual_to_real_state(truncated_end, self._panda_scene_manager)

        is_start_valid, is_goal_valid, sum_free, sum_collision, _ = self._panda_scene_manager.walk_between_waypoints(
            start_, end_)

        is_start_valid = is_start_valid and (truncated_distance_start == 0.0)
        is_goal_valid = is_goal_valid and (truncated_distance_end == 0.0)

        sum_collision += truncated_distance_start + truncated_distance_end
        return start, end, is_start_valid, is_goal_valid, sum_free, sum_collision

    def get_valid_start_goal(self, curriculum_coefficient):
        while True:
            state_size = self._panda_scene_manager.number_of_joints
            virtual_state1 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            if not PandaGameSubgoal.is_free_state_in_manager(virtual_state1, self._panda_scene_manager):
                continue
            virtual_state2 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            if curriculum_coefficient is None:
                # do not use curriculum
                if not PandaGameSubgoal.is_free_state_in_manager(virtual_state2, self._panda_scene_manager):
                    continue
            else:
                # use curriculum
                direction = virtual_state2.copy()
                direction = direction / np.linalg.norm(direction)
                size = np.random.uniform(0., curriculum_coefficient)
                direction *= size
                virtual_state2 = virtual_state1 + direction
            return np.array(virtual_state1), np.array(virtual_state2)
