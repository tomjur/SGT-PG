import numpy as np


class EpisodeRunner:
    def __init__(self, config, game, policy_function):
        self.config = config
        self.game = game
        self.policy_function = policy_function

    def play_random_episodes(self, number_of_episodes, top_level):
        start_goal_pairs = [
            (self.game.get_free_random_state(), self.game.get_free_random_state()) for _ in range(number_of_episodes)
        ]
        return self.play_episodes(start_goal_pairs, top_level)

    def play_episodes(self, start_goal_pairs, top_level):
        # start_goal_pairs = start_goal_pairs * 10
        starts, goals = zip(*start_goal_pairs)
        middle_states = self.policy_function(starts, goals, top_level)
        endpoints = np.array([np.array(starts)] + middle_states + [np.array(goals)])
        endpoints = np.swapaxes(endpoints, 0, 1)
        endpoints = [np.squeeze(e, axis=0) for e in np.vsplit(endpoints, len(endpoints))]
        return {path_id: self._process_endpoints(episode, top_level) for path_id, episode in enumerate(endpoints)}

    def _process_endpoints(self, endpoints, top_level):
        is_valid_episode = True
        base_costs = {}
        splits = {}
        # compute base costs:
        for i in range(len(endpoints)-1):
            start, end = endpoints[i], endpoints[i+1]
            cost, is_valid = self._get_cost(start, end)
            base_costs[(i, i+1)] = (start, end, cost)
            is_valid_episode = is_valid_episode and is_valid

        # compute for the upper levels
        for l in range(1, top_level + 1):
            steps = 2 ** (top_level - l)
            for i in range(steps):
                start_index = i * (2 ** l)
                end_index = (i + 1) * (2**l)
                middle_index = (start_index + end_index) / 2
                start, middle, end = endpoints[start_index], endpoints[middle_index], endpoints[end_index]
                cost_from = splits if l > 1 else base_costs
                cost = cost_from[(start_index, middle_index)][-1] + cost_from[(middle_index, end_index)][-1]
                splits[(start_index, end_index)] = (start, end, middle, cost)
        return endpoints, splits, base_costs, is_valid_episode

    def _get_cost(self, start, goal):
        distance = np.linalg.norm(start - goal)
        max_element = np.max(np.abs(np.concatenate((start, goal), axis=0)))
        if max_element > 1.:
            cost = self.config['cost']['collision_cost'] * (1.+distance)
            cost = cost * cost
            is_valid = False
        else:
            is_valid = self.game.check_terminal_segment((start, goal))
            if is_valid:
                distance_coefficient = self.config['cost']['free_cost']
                cost = distance_coefficient * distance
            else:
                distance_coefficient = self.config['cost']['collision_cost']
                cost = distance_coefficient * (1. + distance)
        return cost, is_valid



