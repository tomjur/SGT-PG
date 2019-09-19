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

    def check_terminal_segment(self, segment):
        is_start_free = self.is_free_state(segment[0])
        is_goal_free = self.is_free_state(segment[1])
        collision_length = self.point_robot_manager.get_collision_length_in_segment(segment[0], segment[1])
        return is_start_free, is_goal_free, collision_length

    def is_free_state(self, state):
        return self.point_robot_manager.is_free(state)

    def _get_state_bounds(self):
        return (-1., -1.), (1., 1.)
