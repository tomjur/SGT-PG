import numpy as np


class EpisodeRunner:
    def __init__(self, config, game, policy_function):
        self.config = config
        self.game = game
        self.policy_function = policy_function
        self.collision_cost = self.config['cost']['collision_cost']
        self.free_cost = self.config['cost']['free_cost']
        self.huber_loss_delta = self.config['cost']['huber_loss_delta']

    def play_random_episodes(self, number_of_episodes, top_level, is_train):
        start_goal_pairs = [
            (self.game.get_free_random_state(), self.game.get_free_random_state()) for _ in range(number_of_episodes)
        ]
        return self.play_episodes(start_goal_pairs, top_level, is_train)

    def play_episodes(self, start_goal_pairs, top_level, is_train):
        starts, goals = zip(*start_goal_pairs)
        middle_states = self.policy_function(starts, goals, top_level, is_train)
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
            cost, is_start_valid, is_goal_valid, is_segment_valid = self._get_cost(start, end)
            base_costs[(i, i+1)] = (start, end, is_start_valid, is_goal_valid, cost)
            is_valid_episode = is_valid_episode and is_segment_valid

        # compute for the upper levels
        for l in range(1, top_level + 1):
            steps = 2 ** (top_level - l)
            splits[l] = {}
            for i in range(steps):
                start_index = i * (2 ** l)
                end_index = (i + 1) * (2**l)
                middle_index = (start_index + end_index) / 2
                start, middle, end = endpoints[start_index], endpoints[middle_index], endpoints[end_index]
                cost_from = splits[l-1] if l > 1 else base_costs
                first_is_start_valid, first_is_goal_valid, first_cost = cost_from[(start_index, middle_index)][-3:]
                second_is_start_valid, second_is_goal_valid, second_cost = cost_from[(middle_index, end_index)][-3:]
                assert first_is_goal_valid == second_is_start_valid
                is_start_valid = first_is_start_valid
                is_goal_valid = second_is_goal_valid
                cost = first_cost + second_cost
                splits[l][(start_index, end_index)] = (start, end, middle, is_start_valid, is_goal_valid, cost)
        return endpoints, splits, base_costs, is_valid_episode

    def _get_cost(self, start, goal):
        distance = np.linalg.norm(start - goal)
        # distance = self._get_huber_loss(distance)
        is_start_valid = self.game.is_free_state(start)
        is_goal_valid = self.game.is_free_state(goal)
        segment_collision = self.game.check_terminal_segment((start, goal))
        is_segment_valid = segment_collision == 0.0

        segment_free = np.maximum(distance - segment_collision, 0.0)

        free_cost = self._get_distance_cost(segment_free) * self.free_cost
        collision_cost = self._get_distance_cost(segment_collision) * self.collision_cost
        cost = free_cost + collision_cost

        return cost, is_start_valid, is_goal_valid, is_segment_valid

    def _get_distance_cost(self, distance):
        if self.config['cost']['type'] == 'linear':
            return distance
        elif self.config['cost']['type'] == 'huber':
            return self._get_huber_loss(distance)

    def _get_huber_loss(self, distance):
        if distance < self.huber_loss_delta:
            return 0.5 * distance * distance
        return self.huber_loss_delta * (distance - 0.5 * self.huber_loss_delta)
