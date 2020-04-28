import os
import numpy as np
import random
import multiprocessing
import queue
import time

from abstract_motion_planning_game_subgoal import AbstractMotionPlanningGameSubgoal
from disks_manager import DisksManager


class DisksSubgoalGame(AbstractMotionPlanningGameSubgoal):
    def __init__(self, max_cores=None, shaping_coeff=0.):
        self.shaping_coeff = shaping_coeff
        self.disks_manager = DisksManager()
        self.pos_size = 2
        self.number_of_disks = 2

        self._hard_grid_states = self._get_hard_grid_states()
        self._all_hard_motions = [(s, g) for s in self._hard_grid_states for g in self._hard_grid_states]

        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()

        self._number_of_workers = self._get_number_of_workers(max_cores)
        self.workers = [
            GameWorker(self.pos_size, self.number_of_disks, self.requests_queue, self.results_queue) for _ in range(self._number_of_workers)
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
    def _get_hard_grid_states():
        grid_states = []
        for y in np.linspace(0.5, 0.7, num=2, endpoint=True):
            for x in np.linspace(-0.7, 0.7, num=8, endpoint=True):
                grid_states.append((x, y))
                grid_states.append((x, -y))
        for y in np.linspace(-0.5, 0.5, num=6, endpoint=True):
            for x in np.linspace(-0.1, 0.1, num=2, endpoint=True):
                grid_states.append((x, y))
        return grid_states

    def get_fixed_start_goal_pairs(self, challenging=False):
        if challenging:
            if self.number_of_disks == 1:
                return [((0.5, 0.5), (-0.5, -0.5))]
            if self.number_of_disks == 2:
                return [((-0.5, -0.5, 0.5, 0.5), (0.5, 0.5, -0.5, -0.5))]
            if self.number_of_disks == 3:
                return [((0., 0., -0.5, -0.5, 0.5, 0.5), (0., 0., 0.5, 0.5, -0.5, -0.5))]
            assert False
        return self._get_all_hard_combinations(self.number_of_disks)

    def _get_all_hard_combinations(self, disks_left):
        if disks_left == 1:
            my_res = self._all_hard_motions
        else:
            next_disk_res = self._get_all_hard_combinations(disks_left-1)
            my_res = [(s + s_n, g + g_n) for s_n, g_n in next_disk_res for s, g in self._all_hard_motions]
            my_res = [(s, g) for s, g in my_res if not self._has_duplicates(s) and not self._has_duplicates(g)]
        random.shuffle(my_res)
        returned_result = []
        for (s, g) in my_res:
            if len(returned_result) == 200:
                break
            if self.is_free_state(s, disks_left) and self.is_free_state(g, disks_left):
                returned_result.append((s, g))
        return returned_result

    def _has_duplicates(self, s):
        pos = self._state_to_poses(s)
        all_p = {tuple(p) for p in pos}
        return len(all_p) < len(pos)

    def get_start_goals(self, number_of_pairs, curriculum_coefficient, get_free_states=True):
        message = (0, get_free_states)
        for _ in range(number_of_pairs):
            self.requests_queue.put(message)

        results = []
        for _ in range(number_of_pairs):
            results.append(self.results_queue.get())
        return results

    def test_predictions(self, cost_queries):
        # put all requests

        requests_count = 0
        for path_id in cost_queries:
            for i, start, end in cost_queries[path_id]:
                message = (1, (path_id, i, start, end))
                requests_count += 1
                self.requests_queue.put(message)

        result = {}
        for _ in range(requests_count):
            path_id, i, start, end, free_cost, collision_cost, minimal_distance = self.results_queue.get()
            minimal_distance_cost = self.shaping_coeff / (1. + minimal_distance)
            free_cost = free_cost + minimal_distance_cost
            if path_id not in result:
                result[path_id] = {}
            result[path_id][i] = (start, end, True, True, free_cost, collision_cost)

        return result

    def is_free_state(self, state, number_of_disks=None):
        disks_pos = self._state_to_poses(state)
        if number_of_disks is None:
            number_of_disks = self.number_of_disks
        assert len(disks_pos) == number_of_disks
        in_place_motion = [[p, p] for p in disks_pos]
        free_area, collision_area, minimal_distance = self.disks_manager.is_motion_free(in_place_motion)
        return collision_area == 0.

    def get_state_size(self):
        return self.pos_size * self.number_of_disks

    def _state_to_poses(self, state):
        return [
            [state[i], state[i+1]] for i in range(0, len(state), self.pos_size)
        ]


class GameWorker(multiprocessing.Process):
    def __init__(self, pos_size, number_of_disks, requests_queue, results_queue):
        multiprocessing.Process.__init__(self)
        self.pos_size = pos_size
        self.number_of_disks = number_of_disks
        self.state_size = self.pos_size * self.number_of_disks
        self.requests_queue = requests_queue
        self.results_queue = results_queue

        self.disks_manager = None
        self.random = None

    def run(self):
        self.random = random.Random(os.getpid())
        self.disks_manager = DisksManager()
        while True:
            try:
                message_type, params = self.requests_queue.get(block=True, timeout=0.001)
                if message_type == 0:
                    # get a valid random state
                    get_free_states = params
                    response = (self._generate_state(get_free_states), self._generate_state(get_free_states))
                elif message_type == 1:
                    # check terminal segments
                    path_id, i, start, end = params
                    free_cost, collision_cost, minimal_distance = self.check_terminal_segment(start, end)
                    response = (path_id, i, start, end, free_cost, collision_cost, minimal_distance)
                else:
                    assert False
                self.results_queue.put(response)
            except queue.Empty:
                time.sleep(1.0)

    def check_terminal_segment(self, start, end):
        segment_length = np.linalg.norm(np.array(start) - np.array(end))
        motions = list(zip(self._state_to_poses(start), self._state_to_poses(end)))
        free_area, collision_area, minimal_distance = self.disks_manager.is_motion_free(motions)
        return free_area * segment_length, collision_area * segment_length, minimal_distance

    def _generate_state(self, test_free):
        state = [self.random.uniform(-1., 1.) for _ in range(self.state_size)]
        if not test_free:
            return state
        while not self.is_free_state(state):
            state = [self.random.uniform(-1., 1.) for _ in range(self.state_size)]
        return state

    def is_free_state(self, state, number_of_disks=None):
        disks_pos = self._state_to_poses(state)
        if number_of_disks is None:
            number_of_disks = self.number_of_disks
        assert len(disks_pos) == number_of_disks
        in_place_motion = [[p, p] for p in disks_pos]
        free_area, collision_area, minimal_distance = self.disks_manager.is_motion_free(in_place_motion)
        return collision_area == 0.

    def _state_to_poses(self, state):
        return [
            [state[i], state[i+1]] for i in range(0, len(state), self.pos_size)
        ]
