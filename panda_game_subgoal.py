import os
import random
from random import Random
import numpy as np
import multiprocessing
import queue
import time

from panda_scene_manager import PandaSceneManager
from abstract_motion_planning_game_subgoal import AbstractMotionPlanningGameSubgoal
from path_helper import get_start_goal_from_scenario


class PandaGameSubgoal(AbstractMotionPlanningGameSubgoal):
    def __init__(self, scenario, max_cores=None, max_queries_buffer_size=1000000, queries_update_freq=2):
        self.scenario = scenario
        self.panda_scene_manager = PandaSceneManager.get_scene_manager(scenario)
        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()

        self._number_of_workers = self._get_number_of_workers(max_cores)
        self.workers = [
            GameWorker(scenario, self.requests_queue, self.results_queue) for _ in range(self._number_of_workers)
        ]

        for w in self.workers:
            w.start()

        self._max_queries_buffer_size = max_queries_buffer_size
        self._queries_update_freq = queries_update_freq
        self._queries_buffer = []
        self._queries_update_counter = 0

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

    def get_fixed_start_goal_pairs(self, challenging=False):
        with open(get_start_goal_from_scenario(self.scenario), 'r') as f:
            lines = [l.replace(os.linesep, '').replace('[', '').replace(']', '') for l in f.readlines()]

        result = []
        while len(result) * 2 < len(lines):
            index = len(result)
            goal = np.array([float(f) for f in lines[2 * index + 1].split(', ')])
            if '_fixed_start' in self.scenario:
                start = np.array([0.0 for _ in range(len(goal))])
            else:
                start = np.array([float(f) for f in lines[2 * index].split(', ')])
            # append to results
            result.append((start, goal))
        if challenging:
            return [result[0]]
        return result

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

    def get_start_goals(self, number_of_episodes, curriculum_coefficient, get_free_states):
        # how many episodes to collect?
        if len(self._queries_buffer) < number_of_episodes:
            # initial steps - collect enough to sample randomly for a couple of cycles
            queries_to_generate = max(number_of_episodes * self._queries_update_freq, self._number_of_workers)
        else:
            if self._queries_update_counter == self._queries_update_freq:
                queries_to_generate = self._number_of_workers
                self._queries_update_counter = 0
            else:
                queries_to_generate = 0
                self._queries_update_counter += 1
        # collect
        new_queries = self._get_start_goals_from_game(queries_to_generate, curriculum_coefficient, get_free_states)
        self._queries_buffer.extend(new_queries)
        # shuffle and remove extra
        random.shuffle(self._queries_buffer)
        self._queries_buffer = self._queries_buffer[:self._max_queries_buffer_size]
        return self._queries_buffer[:number_of_episodes]

    def _get_start_goals_from_game(self, number_of_episodes, curriculum_coefficient, get_free_states):
        # put all requests
        for i in range(number_of_episodes):
            self.requests_queue.put((i, 0, (curriculum_coefficient, get_free_states)))

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
                    curriculum_coefficient, get_free_states = params
                    response = self.get_valid_start_goal(curriculum_coefficient, get_free_states)
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

        self._panda_scene_manager.change_robot_joints(start_)
        sum_free, sum_collision = self._panda_scene_manager.smooth_walk(end_, max_target_distance=1., sensitivity=0.01)

        is_start_valid = truncated_distance_start == 0.0
        is_goal_valid = truncated_distance_end == 0.0

        sum_collision += truncated_distance_start + truncated_distance_end
        return start, end, is_start_valid, is_goal_valid, sum_free, sum_collision

    def get_valid_start_goal(self, curriculum_coefficient, get_free_states):
        while True:
            state_size = self._panda_scene_manager.number_of_joints
            if '_fixed_start' in self.scenario:
                virtual_state1 = [0.0 for _ in range(state_size)]
            else:
                virtual_state1 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            if not PandaGameSubgoal.is_free_state_in_manager(virtual_state1, self._panda_scene_manager):
                continue
            virtual_state2 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            if curriculum_coefficient is not None:
                # use curriculum
                curriculum_coefficient = min(curriculum_coefficient, 6.)
                direction = virtual_state2.copy()
                direction = direction / np.linalg.norm(direction)
                original_size = np.linalg.norm(np.array(virtual_state1) - np.array(virtual_state2))
                size = self._random.uniform(0., min(curriculum_coefficient, original_size))
                direction *= size
                virtual_state2 = virtual_state1 + direction
            if PandaGameSubgoal.is_free_state_in_manager(virtual_state2, self._panda_scene_manager):
                return np.array(virtual_state1), np.array(virtual_state2)
