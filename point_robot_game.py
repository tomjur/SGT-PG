import cPickle as pickle

from point_robot_manager import PointRobotManager
from rl_interface import AbstractMotionPlanningGame


class PointRobotGame(AbstractMotionPlanningGame):
    def __init__(self, config):
        AbstractMotionPlanningGame.__init__(self, config)
        with open(config['general']['params_file'], 'rb') as params_file:
            obstacles_definitions_list = pickle.load(params_file)
        self.point_robot_manager = PointRobotManager(obstacles_definitions_list)

    def check_terminal_segment(self, segment):
        return self.point_robot_manager.is_linear_path_valid(segment[0], segment[1])

    def is_free_state(self, state):
        return self.point_robot_manager.is_free(state)
