import numpy as np
import pickle

from abstract_motion_planning_game_sequential import AbstractMotionPlanningGameSequential
from path_helper import get_params_from_scenario
from point_robot_manager import PointRobotManager


class PointRobotGameSequential(AbstractMotionPlanningGameSequential):
    def __init__(self, scenario, collision_cost, goal_reward, max_action_limit=None, max_steps=None,
                 goal_closeness_distance=0.01):
        params_file = get_params_from_scenario(scenario)
        if 'no_obs' in params_file:
            obstacles_definitions_list = []
        else:
            with open(params_file, 'rb') as f:
                obstacles_definitions_list = pickle.load(f)
        self.point_robot_manager = PointRobotManager(obstacles_definitions_list)
        self.max_steps = max_steps
        self.collision_cost = collision_cost
        self.goal_reward = goal_reward
        self.max_action_limit = max_action_limit  # if None, there is no limit and the robot can just teleport
        self.goal_closeness_distance = goal_closeness_distance

    def _get_random_state(self):
        dim = self.point_robot_manager.dimension_length
        return np.random.uniform(-dim, dim, 2)

    def _get_free_state(self):
        while True:
            state = self._get_random_state()
            if self.point_robot_manager.is_free(state):
                return state

    def _are_close(self, s1, s2):
        return np.linalg.norm(np.array(s1) - np.array(s2)) < self.goal_closeness_distance

    def get_free_start_goals(self, number_of_episodes, curriculum_coefficient):
        result = []
        while len(result) < number_of_episodes:
            s = self._get_free_state()
            if curriculum_coefficient is None:
                # don't use curriculum, get a free state
                g = self._get_free_state()
                if not self._are_close(s, g):
                    result.append((s, g))
            else:
                # use curriculum, choose a direction vector, and advance according to the direction
                assert curriculum_coefficient > 1.0
                direction = self._get_random_state()
                direction = direction / np.linalg.norm(direction)
                direction *= self.goal_closeness_distance
                size = np.random.uniform(1., curriculum_coefficient)
                direction *= size
                g = s + direction
                if self.is_free_state(g):
                    result.append((s, g))
        return result

    def is_free_state(self, state):
        return self.point_robot_manager.is_free(state)

    def get_fixed_start_goal_pairs(self):
        return self.point_robot_manager.get_fixed_start_goal_pairs()

    def run_episodes(self, start_goal_pairs, is_train, policy_function):
        active_pairs = {path_id: pair for path_id, pair in enumerate(start_goal_pairs)}
        states = {path_id: [active_pairs[path_id][0]] for path_id in active_pairs}
        goals = {path_id: active_pairs[path_id][1] for path_id in active_pairs}
        successes = {path_id: False for path_id in active_pairs}
        actions = {path_id: [] for path_id in active_pairs}
        costs = {path_id: [] for path_id in active_pairs}
        while len(active_pairs) > 0:
            # create predictions for all active pairs
            active_path_ids = list(active_pairs.keys())
            active_current_states = [states[path_id][-1] for path_id in active_path_ids]
            active_goal_states = [goals[path_id] for path_id in active_path_ids]
            active_actions = policy_function(active_current_states, active_goal_states, is_train)
            # act according to prediction
            for i, path_id in enumerate(active_path_ids):
                state = active_current_states[i]
                action = active_actions[i]
                next_state = state + self._get_actual_action(action)
                free_length, collision_length = self.point_robot_manager.get_collision_length_in_segment(
                    state, next_state)
                at_goal = self._are_close(goals[path_id], next_state)
                cost = self._get_cost(free_length, collision_length, at_goal)
                # if done remove from active
                is_collision = collision_length > 0.
                # update the results
                states[path_id].append(next_state)
                actions[path_id].append(action)
                costs[path_id].append(cost)
                if at_goal and not is_collision:
                    successes[path_id] = True

                max_steps_reached = False
                if self.max_steps is not None:
                    max_steps_reached = len(actions[path_id]) == self.max_steps
                done = at_goal or is_collision or max_steps_reached
                if done:
                    active_pairs.pop(path_id)
        results = {
            path_id: (states[path_id], goals[path_id], actions[path_id], costs[path_id], successes[path_id])
            for path_id in states
        }
        return results

    def _get_actual_action(self, action):
        action_ = action.copy()
        if self.max_action_limit is not None:
            action_size = np.linalg.norm(action_)
            # walk in a limited space
            if action_size > self.max_action_limit:
                action_ /= action_size
                action_ *= self.max_action_limit
        return action_

    def _get_cost(self, free_length, collision_length, are_close):
        if collision_length > 0.:
            return free_length + self.collision_cost * collision_length
        else:
            return free_length - self.goal_reward * are_close

    def get_state_space_size(self):
        return 2

    def get_action_space_size(self):
        return 2
