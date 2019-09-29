import numpy as np
import cPickle as pickle

from point_robot_manager import PointRobotManager
from rl_interface import AbstractMotionPlanningGame


class PointRobotGame(AbstractMotionPlanningGame):
    def __init__(self, config):
        AbstractMotionPlanningGame.__init__(self, config)
        if 'no_obs' in self.params_file:
            obstacles_definitions_list = []
        else:
            with open(self.params_file, 'rb') as params_file:
                obstacles_definitions_list = pickle.load(params_file)
        self.point_robot_manager = PointRobotManager(obstacles_definitions_list)

    def can_recover_from_failed_movement(self):
        # point robot can teleport between states
        return True

    def check_terminal_segment(self, segment):
        is_start_free = self.is_free_state(segment[0])
        is_goal_free = self.is_free_state(segment[1])
        collision_length = self.point_robot_manager.get_collision_length_in_segment(segment[0], segment[1])
        return is_start_free, is_goal_free, collision_length

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
