import pickle
import numpy as np

from path_helper import get_params_from_scenario
from point_robot_manager import PointRobotManager
from abstract_motion_planning_game_subgoal import AbstractMotionPlanningGameSubgoal


class PointRobotGameSubgoal(AbstractMotionPlanningGameSubgoal):
    def __init__(self, scenario):
        params_file = get_params_from_scenario(scenario)
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

    def _get_random_state(self):
        dim = self.point_robot_manager.dimension_length
        return np.random.uniform(-dim, dim, self.get_state_size())

    def _get_free_state(self):
        while True:
            state = self._get_random_state()
            if self.point_robot_manager.is_free(state):
                return state

    def get_start_goals(self, number_of_episodes, curriculum_coefficient, get_free_states):
        result = []
        while len(result) < number_of_episodes:
            s = self._get_free_state() if get_free_states else self._get_random_state()
            if curriculum_coefficient is None:
                # don't use curriculum, get a free state
                g = self._get_free_state() if get_free_states else self._get_random_state()
                result.append((s, g))
            else:
                # use curriculum, choose a direction vector, and advance according to the direction
                direction = self._get_random_state()
                direction = direction / np.linalg.norm(direction)
                size = np.random.uniform(0., curriculum_coefficient)
                direction *= size
                g = s + direction
                if not get_free_states or self.is_free_state(g):
                    result.append((s, g))
        return result

    def is_free_state(self, state):
        return self.point_robot_manager.is_free(state)

    def get_fixed_start_goal_pairs(self, challenging=False):
        if challenging:
            return [(np.array((-0.9, -0.9)), np.array((0.9, 0.9)))]
        return self.point_robot_manager.get_fixed_start_goal_pairs()

    def get_state_size(self):
        return 2
