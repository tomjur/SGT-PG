import os
import random
import numpy as np

from docker_path_helper import init_dir
from log_utils import print_and_log


class Trainer:
    def __init__(self,  model_name, config, working_dir, network, sess, episode_runner, summaries_collector):
        self.model_name = model_name
        self.config = config
        self.working_dir = working_dir
        self.network = network
        self.sess = sess
        self.episode_runner = episode_runner
        self.summaries_collector = summaries_collector

        self.summaries_frequency = config['general']['write_summaries_every']
        self.batch_size = config['model']['batch_size']
        self.steps_per_trajectory_print = config['general']['cycles_per_trajectory_print']
        self.train_levels = config['model']['train_levels']
        self.train_episodes_per_cycle = config['general']['train_episodes_per_cycle']
        self.gain = config['model']['gain']

    def train_policy_at_level(self, top_level, global_step):
        successes, accumulated_cost, dataset = self.collect_data(
            self.train_episodes_per_cycle, top_level, trajectories_dir=self._get_trajectories_dir(global_step),
            is_train=True, use_fixed_start_goal_pairs=False)
        if global_step % self.summaries_frequency == 0:
            self.summaries_collector.write_train_success_summaries(self.sess, global_step, successes, accumulated_cost)

        for level in self._get_relevant_levels(top_level):
            valid_data = [
                (s, g, m, c) for (s, g, m, s_valid, g_valid, c) in dataset[level] if s_valid and g_valid
            ]
            if len(valid_data) == 0:
                continue
            starts, ends, middles, costs = zip(*random.sample(valid_data, min(self.batch_size, len(valid_data))))
            baseline_costs = self.network.predict_value(starts, ends, level, self.sess)
            initial_costs = np.expand_dims(np.array(costs), axis=-1)
            costs = initial_costs - baseline_costs
            summaries, prediction_loss, _ = self.network.train_policy(level, starts, ends, middles, costs, self.sess)
            if global_step % self.summaries_frequency == 0:
                self.summaries_collector.write_train_optimization_summaries(summaries, global_step)
            global_step += 1
        return global_step

    def train_value_function_at_level(self, top_level, global_step):
        successes, accumulated_cost, dataset = self.collect_data(
            self.train_episodes_per_cycle, top_level, trajectories_dir=self._get_trajectories_dir(global_step),
            is_train=True, use_fixed_start_goal_pairs=False)
        if global_step % self.summaries_frequency == 0:
            self.summaries_collector.write_train_success_summaries(self.sess, global_step, successes, accumulated_cost)

        prediction_loss = None
        for level in self._get_relevant_levels(top_level):
            valid_data = [
                (s, g, m, c) for (s, g, m, s_valid, g_valid, c) in dataset[level] if s_valid and g_valid
            ]
            if len(valid_data) == 0:
                continue
            starts, ends, _, costs = zip(*random.sample(valid_data, min(self.batch_size, len(valid_data))))

            summaries, prediction_loss, _ = self.network.train_value(level, starts, ends, costs, self.sess)
            if global_step % self.summaries_frequency == 0:
                self.summaries_collector.write_train_optimization_summaries(summaries, global_step)
            global_step += 1
        return global_step, prediction_loss

    def collect_data(self, count, top_level, trajectories_dir=None, is_train=True, use_fixed_start_goal_pairs=False):
        print_and_log('collecting {} {} episodes of level {}'.format(count, 'train' if is_train else 'test', top_level))
        if use_fixed_start_goal_pairs:
            episode_results = self.episode_runner.play_fixed_episodes(top_level, is_train)
        else:
            episode_results = self.episode_runner.play_random_episodes(count, top_level, is_train)
        accumulated_cost, successes = [], []
        dataset = {level: [] for level in range(1, top_level + 1)}

        if trajectories_dir is not None:
            init_dir(trajectories_dir)

        for path_id in episode_results:
            endpoints, splits, base_costs, is_valid_episode = episode_results[path_id]

            # write to file if needed
            if trajectories_dir is not None:
                if is_valid_episode:
                    trajectory_filename = '{}_success.txt'.format(path_id)
                else:
                    trajectory_filename = '{}_collision.txt'.format(path_id)

                with open(os.path.join(trajectories_dir, trajectory_filename), 'w') as f:
                    for e in endpoints:
                        f.write('{}{}'.format(str(e), os.linesep))

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
        print_and_log(
            'data collection done, success rate is {}, accumulated cost is {}'.format(successes, accumulated_cost))
        return successes, accumulated_cost, dataset

    def _get_trajectories_dir(self, global_step):
        if global_step % self.steps_per_trajectory_print == 0:
            return os.path.join(self.working_dir, 'trajectories', self.model_name, str(global_step))
        else:
            return None

    def _get_relevant_levels(self, top_level):
        if self.train_levels == 'all-below':
            return range(1, top_level + 1)
        elif self.train_levels == 'topmost':
            return range(top_level, top_level + 1)
        else:
            assert False
