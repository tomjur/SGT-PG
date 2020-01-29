import os
import random
import numpy as np
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from path_helper import init_dir
from log_utils import print_and_log
from model_saver import ModelSaver


class TrainerSubgoalSequential:
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

    def train_policy_at_level(self, global_step):
        successes, accumulated_cost, dataset, endpoints_by_path = self.collect_train_data(self.train_episodes_per_cycle)
        self.train_episodes_counter += len(endpoints_by_path)
        self.summaries_collector.write_train_success_summaries(
            self.sess, global_step, successes, accumulated_cost, self.curriculum_coefficient)

        # set the baseline to the current policy
        self.network.update_baseline_policy(self.sess)
        # do optimization steps
        for update_step in range(self.config['model']['consecutive_optimization_steps']):
            starts, ends, middles, costs = zip(*random.sample(dataset, min(self.batch_size, len(dataset))))
            costs = costs = np.expand_dims(np.array(costs), axis=-1)
            try:
                initial_gradient_norm, _, summaries, prediction_loss, _ = self.network.train_policy(
                    starts, ends, middles, costs, self.sess
                )
                self.summaries_collector.write_train_optimization_summaries(summaries, global_step)
                global_step += 1
            except InvalidArgumentError as error:
                print('error encountered')
                break

        return global_step, successes

    def collect_train_data(self, count):
        print_and_log('collecting {} train episodes'.format(count))
        start_goal_pairs = self.episode_runner.game.get_free_start_goals(count, self.curriculum_coefficient)
        return self.collect_data(start_goal_pairs, True)

    def collect_test_data(self, is_challenging=False):
        if is_challenging:
            start_goal_pairs = self.hard_fixed_start_goal_pairs
        else:
            start_goal_pairs = self.fixed_start_goal_pairs
        print_and_log('collecting {} test episodes'.format(len(start_goal_pairs)))
        return self.collect_data(start_goal_pairs, False)

    def collect_data(self, start_goal_pairs, is_train):
        episode_results = self.episode_runner.play_episodes(start_goal_pairs, is_train)
        successes, accumulated_cost, dataset, endpoints_by_path = self._process_episode_results(episode_results)
        print_and_log(
            'data collection done, success rate is {}, accumulated cost is {}'.format(successes, accumulated_cost))
        return successes, accumulated_cost, dataset, endpoints_by_path

    def _process_episode_results(self, episode_results):
        accumulated_cost, successes = [], []
        endpoints_by_path = {}
        dataset = []

        for path_id in episode_results:
            endpoints, base_costs, future_costs, is_valid_episode = episode_results[path_id]
            goal = endpoints[-1]
            # add data
            for start_state, next_state, _, _, cost in future_costs:
                dataset.append((start_state, goal, next_state, cost))

            endpoints_by_path[path_id] = (endpoints, is_valid_episode)

            # log collision
            successes.append(is_valid_episode)

            # total cost:
            total_cost = future_costs[-1][-1]
            accumulated_cost.append(total_cost)

        successes = np.mean(successes)
        accumulated_cost = np.mean(accumulated_cost)
        return successes, accumulated_cost, dataset, endpoints_by_path
