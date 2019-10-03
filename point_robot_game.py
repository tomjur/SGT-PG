import numpy as np
import cPickle as pickle

from point_robot_manager import PointRobotManager
from rl_interface import AbstractMotionPlanningGame


class PointRobotGame(AbstractMotionPlanningGame):
    def __init__(self, config):
        AbstractMotionPlanningGame.__init__(self, config)
        params_file = self.get_params_from_config(config)
        if 'no_obs' in params_file:
            obstacles_definitions_list = []
        else:
            with open(params_file, 'rb') as f:
                obstacles_definitions_list = pickle.load(f)
        self.point_robot_manager = PointRobotManager(obstacles_definitions_list)

    def check_terminal_segments(self, cost_queries):
        results = {}
        for path_id in cost_queries:
            results[path_id] = {}
            for i, start, goal in cost_queries[path_id]:
                query_results = self._check_terminal_segment(start, goal)
                results[path_id][i] = query_results
        return results

    def _check_terminal_segment(self, start, goal):
        is_start_free = self.is_free_state(start)
        is_goal_free = self.is_free_state(goal)
        free_length, collision_length = self.point_robot_manager.get_collision_length_in_segment(start, goal)
        return start, goal, is_start_free, is_goal_free, free_length, collision_length

    def is_free_state(self, state):
        return self.point_robot_manager.is_free(state)

    def _get_state_bounds(self):
        return (-1., -1.), (1., 1.)

    def get_fixed_start_goal_pairs(self):
        lower = self.lower
        upper = self.upper
        assert len(lower) == len(upper)
        all_pairs = []
        grid_marks = 11
        while len(all_pairs) < 1000:
            grid_states = self._rec_all_states(0, grid_marks)
            grid_states = [s for s in grid_states if self.is_free_state(s)]
            all_pairs = [(s1, s2) for s1 in grid_states for s2 in grid_states]
            grid_marks += 1
        return all_pairs

    def _rec_all_states(self, state_index, grid_marks):
        s = np.linspace(self.lower[state_index], self.upper[state_index], grid_marks)
        if state_index == len(self.lower) - 1:
            return [[x] for x in s]
        next_res = self._rec_all_states(state_index + 1, grid_marks)
        return [[x] + l[:] for l in next_res for x in s]
