import os
import random
import numpy as np
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from path_helper import init_dir
from log_utils import print_and_log
from model_saver import ModelSaver


class TrainerSubgoal:
    def __init__(self,  model_name, config, working_dir, network, sess, episode_runner, summaries_collector,
                 curriculum_coefficient=None):
        self.model_name = model_name
        self.config = config
        self.working_dir = working_dir
        self.network = network
        self.sess = sess
        self.episode_runner = episode_runner
        self.summaries_collector = summaries_collector
        self.curriculum_coefficient = curriculum_coefficient

        self.fixed_start_goal_pairs = self.episode_runner.game.get_fixed_start_goal_pairs(challenging=False)
        self.hard_fixed_start_goal_pairs = self.episode_runner.game.get_fixed_start_goal_pairs(challenging=True)

        self.batch_size = config['model']['batch_size']
        self.steps_per_trajectory_print = config['general']['cycles_per_trajectory_print']
        self.train_episodes_per_cycle = config['general']['train_episodes_per_cycle']
        self.gain = config['model']['gain']

        self.train_episodes_counter = 0

        self.check_gradients = config['gradient_checker']['enable']
        if self.check_gradients:
            self.gradient_output_dir = os.path.join(working_dir, 'gradient', model_name)
            init_dir(self.gradient_output_dir)
            saver_dir = os.path.join(self.gradient_output_dir, 'temp_4_gradient_print')
            self.gradient_saver = ModelSaver(saver_dir, 1, 'gradient_checker', print_log=False)
        else:
            self.gradient_output_dir, self.gradient_saver = None, None

    @staticmethod
    def _reduce_mean_by_start_goal(starts, ends, costs):
        # compute keys
        keys = [(tuple(starts[i].tolist()), tuple(ends[i].tolist())) for i in range(len(starts))]
        # put all in buckets
        groups = {}
        for i in range(len(costs)):
            key = keys[i]
            cost = costs[i]
            if key in groups:
                groups[key].append(cost)
            else:
                groups[key] = [cost]
        # compute the mean
        mean_groups = {key: np.mean(groups[key]) for key in groups}
        # compute the new costs
        new_costs = [costs[i] - mean_groups[keys[i]] for i in range(len(costs))]
        return new_costs

    def train_policy_at_level(self, top_level, global_step):
        successes, accumulated_cost, dataset, endpoints_by_path = self.collect_train_data(
            self.train_episodes_per_cycle, top_level)
        self.train_episodes_counter += len(endpoints_by_path)
        self.summaries_collector.write_train_success_summaries(
            self.sess, global_step, successes, accumulated_cost, self.curriculum_coefficient)

        valid_data = [
            (s, g, m, c) for (s, g, m, s_valid, g_valid, c) in dataset[top_level] if s_valid and g_valid
        ]
        assert len(valid_data) == len(dataset[top_level])

        # compute the costs now
        starts, ends, middles, costs = zip(*valid_data)
        costs = self._process_costs(starts, ends, costs)
        valid_data = list(zip(starts, ends, middles, costs))

        # set the baseline to the current policy
        self.network.update_baseline_policy(self.sess, top_level)
        # do optimization steps
        for update_step in range(self.config['model']['consecutive_optimization_steps']):
            starts, ends, middles, costs = zip(*random.sample(valid_data, min(self.batch_size, len(valid_data))))
            try:
                initial_gradient_norm, _, summaries, prediction_loss, _ = self.network.train_policy(
                    top_level, starts, ends, middles, costs, self.sess
                )
                self.summaries_collector.write_train_optimization_summaries(summaries, global_step)
                global_step += 1
            except InvalidArgumentError as error:
                print('error encountered')
                break

        return global_step, successes

    def _process_costs(self, starts, ends, costs):
        if self.config['model']['repeat_train_trajectories'] > 0:
            costs = self._reduce_mean_by_start_goal(starts, ends, costs)
        costs = np.expand_dims(np.array(costs), axis=-1)

        return costs

    def collect_train_data(self, count, top_level):
        print_and_log('collecting {} train episodes of level {}'.format(count, top_level))
        start_goal_pairs = self.episode_runner.game.get_start_goals(
            count, self.curriculum_coefficient, get_free_states=True)
        return self.collect_data(start_goal_pairs, top_level, True)

    def collect_test_data(self, top_level, is_challenging=False):
        if is_challenging:
            start_goal_pairs = self.hard_fixed_start_goal_pairs
        else:
            start_goal_pairs = self.fixed_start_goal_pairs
        print_and_log('collecting {} test episodes of level {}'.format(len(start_goal_pairs), top_level))
        return self.collect_data(start_goal_pairs, top_level, False)

    def collect_data(self, start_goal_pairs, top_level, is_train):
        episode_results = self.episode_runner.play_episodes(start_goal_pairs, top_level, is_train)
        successes, accumulated_cost, dataset, endpoints_by_path = self._process_episode_results(
            episode_results, top_level)
        print_and_log(
            'data collection done, success rate is {}, accumulated cost is {}'.format(successes, accumulated_cost))
        return successes, accumulated_cost, dataset, endpoints_by_path

    def _process_episode_results(self, episode_results, top_level):
        accumulated_cost, successes = [], []
        dataset = {level: [] for level in range(1, top_level + 1)}
        endpoints_by_path = {}

        for path_id in episode_results:
            endpoints, splits, base_costs, is_valid_episode = episode_results[path_id]

            endpoints_by_path[path_id] = (endpoints, is_valid_episode)

            # log collision
            successes.append(is_valid_episode)

            # total cost:
            total_cost = splits[top_level][(0, len(endpoints) - 1)][-1]
            accumulated_cost.append(total_cost)

            # extend the dataset
            for level in range(1, top_level + 1):
                current_dataset = splits[level].values()
                if self.gain == 'full-traj':
                    current_dataset = [
                        (s, g, m, s_valid, g_valid, total_cost) for (s, g, m, s_valid, g_valid, c) in current_dataset
                    ]
                elif self.gain == 'future-only':
                    pass
                else:
                    assert False
                dataset[level].extend(current_dataset)

        successes = np.mean(successes)
        accumulated_cost = np.mean(accumulated_cost)
        return successes, accumulated_cost, dataset, endpoints_by_path

