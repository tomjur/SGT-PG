import numpy as np


class EpisodeRunnerSubgoal:
    def __init__(self, config, game, policy_function):
        self.config = config
        self.game = game
        self.policy_function = policy_function

        self.collision_cost = self.config['cost']['collision_cost']
        self.is_constant_collision_cost = self.config['cost']['is_constant_collision_cost']
        self.free_cost = self.config['cost']['free_cost']
        self.is_constant_free_cost = self.config['cost']['is_constant_free_cost']
        self.huber_loss_delta = self.config['cost']['huber_loss_delta']

        self.repeat_train_trajectories = 0
        if 'model' in config and 'repeat_train_trajectories' in config['model']:
            self.repeat_train_trajectories = config['model']['repeat_train_trajectories']

    def play_episodes(self, start_goal_pairs, top_level, is_train):
        if is_train and self.repeat_train_trajectories:
            start_goal_pairs_ = []
            for _ in range(self.repeat_train_trajectories):
                for s, g in start_goal_pairs:
                    new_pair = (s.copy(), g.copy())
                    start_goal_pairs_.append(new_pair)
            start_goal_pairs = start_goal_pairs_
        starts, goals = zip(*start_goal_pairs)
        if top_level > 0:
            middle_states = self.policy_function(starts, goals, top_level, is_train)
            endpoints = np.array([np.array(starts)] + middle_states + [np.array(goals)])
        else:
            endpoints = np.array([np.array(starts)] + [np.array(goals)])
        endpoints = np.swapaxes(endpoints, 0, 1)
        endpoints = [np.squeeze(e, axis=0) for e in np.vsplit(endpoints, len(endpoints))]

        results = {}
        all_costs_queries = {}
        for path_id, episode in enumerate(endpoints):
            results[path_id] = episode
            cost_queries = [(i, episode[i], episode[i+1]) for i in range(len(episode)-1)]
            all_costs_queries[path_id] = cost_queries

        all_cost_responses = self.game.test_predictions(all_costs_queries)

        for path_id in results:
            episode = results[path_id]
            episode_cost_responses = all_cost_responses[path_id]
            results[path_id] = self._process_endpoints(episode, episode_cost_responses, top_level)
        return results

    def _process_endpoints(self, endpoints, cost_responses, top_level):
        is_valid_episode = True
        base_costs = {}
        splits = {}

        # compute base costs:
        for i in range(len(endpoints)-1):
            start, end = endpoints[i], endpoints[i+1]
            cost_response = cost_responses[i]
            assert all(np.equal(start, cost_response[0])), 'i {} start {} cost_response[0] {} endpoints {}'.format(
                i, start, cost_response[0], endpoints)
            assert all(np.equal(end, cost_response[1])), 'i {} end {} cost_response[1] {} endpoints {}'.format(
                i, end, cost_response[1], endpoints)
            is_start_valid, is_goal_valid, free_length, collision_length = cost_response[2:]
            is_segment_valid = collision_length == 0.0
            cost = self._get_cost(free_length, collision_length)
            base_costs[(i, i+1)] = (start, end, is_start_valid, is_goal_valid, cost)
            is_valid_episode = is_valid_episode and is_segment_valid

        # compute for the upper levels
        for l in range(1, top_level + 1):
            steps = 2 ** (top_level - l)
            splits[l] = {}
            for i in range(steps):
                start_index = i * (2 ** l)
                end_index = (i + 1) * (2**l)
                middle_index = int((start_index + end_index) / 2)
                start, middle, end = endpoints[start_index], endpoints[middle_index], endpoints[end_index]
                cost_from = splits[l-1] if l > 1 else base_costs
                first_is_start_valid, first_is_goal_valid, first_cost = cost_from[(start_index, middle_index)][-3:]
                second_is_start_valid, second_is_goal_valid, second_cost = cost_from[(middle_index, end_index)][-3:]
                if first_cost is None or second_cost is None:
                    # if any segment is bad, ignore upper levels
                    is_start_valid, is_goal_valid, cost = None, None, None
                else:
                    # if first_is_goal_valid != second_is_start_valid:
                    #     print_and_log('bad segment agreement')
                    #     print_and_log('start {} middle {} end {}'.format(start, middle, end))
                    #     print_and_log('first_is_goal_valid {} second_is_start_valid {}'.format(first_is_goal_valid, second_is_start_valid))
                    #     assert False
                    is_start_valid = first_is_start_valid
                    is_goal_valid = second_is_goal_valid
                    cost = first_cost + second_cost
                splits[l][(start_index, end_index)] = (start, end, middle, is_start_valid, is_goal_valid, cost)

        return endpoints, splits, base_costs, is_valid_episode

    def _get_cost(self, segment_free, segment_collision):
        if segment_collision == 0.0:
            # segment is collision free
            if self.is_constant_free_cost:
                # pay a fixed cost for not being in collision
                return self.free_cost
            else:
                # pay a distance relative cost for the free segment
                return self._get_distance_cost(segment_free) * self.free_cost
        else:
            # segment in collision
            if self.is_constant_collision_cost:
                # pay a fixed cost for being in collision
                return self.collision_cost
            else:
                # pay a distance relative cost for the free segment and for in-collision segments
                free_cost = self._get_distance_cost(segment_free) * self.free_cost
                collision_cost = self._get_distance_cost(segment_collision) * self.collision_cost
                cost = free_cost + collision_cost
                return cost

    def _get_distance_cost(self, distance):
        if self.config['cost']['type'] == 'linear':
            return distance
        elif self.config['cost']['type'] == 'huber':
            return self._get_huber_loss(distance)
        elif self.config['cost']['type'] == 'square':
            return np.square(distance)

    def _get_huber_loss(self, distance):
        if distance < self.huber_loss_delta:
            return 0.5 * distance * distance
        return self.huber_loss_delta * (distance - 0.5 * self.huber_loss_delta)
