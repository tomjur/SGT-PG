import pickle

from path_helper import get_params_from_scenario
from point_robot_manager import PointRobotManager
from abstract_motion_planning_game_subgoal import AbstractMotionPlanningGameSubgoal


class PointRobotGameSubgoal(AbstractMotionPlanningGameSubgoal):
    def __init__(self, config):
        AbstractMotionPlanningGameSubgoal.__init__(self, config)
        params_file = get_params_from_scenario(config['general']['scenario'])
        if 'no_obs' in params_file:
            obstacles_definitions_list = []
        else:
            with open(params_file, 'rb') as f:
                obstacles_definitions_list = pickle.load(f)
        self.point_robot_manager = PointRobotManager(obstacles_definitions_list)

    def test_predictions(self, cost_queries):
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

    def get_free_states(self, number_of_states):
        result = []
        for _ in range(number_of_states):
            s = self.get_random_state()
            while not self.is_free_state(s):
                s = self.get_random_state()
            result.append(s)
        return result

    def is_free_state(self, state):
        return self.point_robot_manager.is_free(state)

    def _get_state_bounds(self):
        return (-1., -1.), (1., 1.)

    def get_fixed_start_goal_pairs(self):
        return self.point_robot_manager.get_fixed_start_goal_pairs()
