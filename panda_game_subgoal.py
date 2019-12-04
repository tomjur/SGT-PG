import os
from random import Random
import numpy as np
import multiprocessing
import queue
import time

from panda_scene_manager import PandaSceneManager
from abstract_motion_planning_game_subgoal import AbstractMotionPlanningGameSubgoal


class PandaGameSubgoal(AbstractMotionPlanningGameSubgoal):
    def __init__(self, config):
        self.panda_scene_manager = PandaSceneManager.get_scene_manager(config)
        AbstractMotionPlanningGameSubgoal.__init__(self, config)

        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()

        self.workers = [
            GameWorker(config, self.requests_queue, self.results_queue)
            for _ in range(self._get_number_of_workers(config))
        ]

        for w in self.workers:
            w.start()

    @staticmethod
    def _get_number_of_workers(config):
        # return 1  # for debug: set one worker
        # get configuration values
        test_episodes = config['general']['test_episodes']
        train_episodes_per_cycle = config['general']['train_episodes_per_cycle']
        repeat_train_trajectories = config['model']['repeat_train_trajectories']
        max_episodes = max(test_episodes, train_episodes_per_cycle * repeat_train_trajectories)
        # get available cpus
        cores = multiprocessing.cpu_count()
        max_cores_with_slack = max(cores - 2, 1)
        # the number of workers is the min between those
        return min(max_episodes, max_cores_with_slack)


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

    def _get_state_bounds(self):
        lower = np.array([-1.] * self.panda_scene_manager.number_of_joints)
        upper = np.array([1.] * self.panda_scene_manager.number_of_joints)
        return lower, upper

    def get_fixed_start_goal_pairs(self):
        # for now a long straight movement
        lower = np.array(self.panda_scene_manager.joints_lower_bounds).copy()
        upper = np.array(self.panda_scene_manager.joints_upper_bounds).copy()

        scenario = self.config['general']['scenario']
        if scenario == 'panda_no_obs':
            s = self.real_to_virtual_state(0.5 * upper + 0.5 * lower, self.panda_scene_manager)
            g = self.real_to_virtual_state(upper.copy(), self.panda_scene_manager)
        elif scenario == 'panda_easy':
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
        elif scenario == 'panda_hard':
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
            path_requests = (path_id, tuple(cost_queries[path_id]))
            self.requests_queue.put(path_requests)

        # pull all responses
        results = {}
        for _ in range(len(cost_queries)):
            response = self.results_queue.get(block=True)
            path_id, path_results = response
            results[path_id] = {t[0]: t[1] for t in path_results}
        return results

    def get_free_states(self, number_of_states):
        # put all requests
        for i in range(number_of_states):
            self.requests_queue.put((i, None))

        # pull all responses
        results = []
        for _ in range(number_of_states):
            response = self.results_queue.get(block=True)
            results.append(response)
        return results

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
    def __init__(self, config, requests_queue, results_queue):
        multiprocessing.Process.__init__(self)
        self.config = config
        self.requests_queue = requests_queue
        self.results_queue = results_queue

        self.panda_scene_manager = None
        self.random = None

    def run(self):
        self.random = Random(os.getpid())
        self.panda_scene_manager = PandaSceneManager.get_scene_manager(self.config)
        while True:
            try:
                request = self.requests_queue.get(block=True, timeout=0.001)
                path_id, cost_queries = request
                if cost_queries is None:
                    # get a valid random state
                    response = self.get_valid_state()
                else:
                    # check terminal segments
                    path_cost_results = [
                        (i, self.check_terminal_segment(start, end))
                        for i, start, end in cost_queries
                    ]
                    response = (path_id, path_cost_results)
                self.results_queue.put(response)
            except queue.Empty:
                time.sleep(1.0)

    def check_terminal_segment(self, start, end):
        self.panda_scene_manager.reset_simulation()
        # check the end points
        truncated_start, truncated_distance_start = PandaGameSubgoal._truncate_virtual_state(start)
        truncated_end, truncated_distance_end = PandaGameSubgoal._truncate_virtual_state(end)

        start_ = PandaGameSubgoal.virtual_to_real_state(truncated_start, self.panda_scene_manager)
        end_ = PandaGameSubgoal.virtual_to_real_state(truncated_end, self.panda_scene_manager)

        is_start_valid, is_goal_valid, sum_free, sum_collision, _ = self.panda_scene_manager.walk_between_waypoints(
            start_, end_)

        is_start_valid = is_start_valid and (truncated_distance_start == 0.0)
        is_goal_valid = is_goal_valid and (truncated_distance_end == 0.0)

        sum_collision += truncated_distance_start + truncated_distance_end
        return start, end, is_start_valid, is_goal_valid, sum_free, sum_collision

    def get_valid_state(self):
        while True:
            state_size = len(self.panda_scene_manager.joints_lower_bounds)
            virtual_state = [self.random.uniform(-1., 1.) for _ in range(state_size)]
            if PandaGameSubgoal.is_free_state_in_manager(virtual_state, self.panda_scene_manager):
                return virtual_state
