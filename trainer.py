import os
import random
import numpy as np

from path_helper import init_dir
from log_utils import print_and_log
from model_saver import ModelSaver


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

        self.train_trajectories_dir = os.path.join(self.working_dir, 'trajectories', self.model_name)
        init_dir(self.train_trajectories_dir)

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
        successes, accumulated_cost, dataset = self.collect_data(
            self.train_episodes_per_cycle, top_level, trajectories_file=self._get_trajectories_file(global_step),
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
            costs = self._process_costs(starts, ends, costs, level)
            summaries, prediction_loss, _ = self.network.train_policy(level, starts, ends, middles, costs, self.sess)
            if global_step % self.summaries_frequency == 0:
                self.summaries_collector.write_train_optimization_summaries(summaries, global_step)
            global_step += 1
        return global_step

    def _process_costs(self, starts, ends, costs, level):
        if self.config['model']['repeat_train_trajectories'] > 0:
            costs = self._reduce_mean_by_start_goal(starts, ends, costs)
        costs = np.expand_dims(np.array(costs), axis=-1)
        if self.config['value_function']['use_value_functions']:
            baseline_costs = self.network.predict_value(starts, ends, level, self.sess)
            costs = costs - baseline_costs
        return costs

    def train_value_function_at_level(self, top_level, global_step):
        successes, accumulated_cost, dataset = self.collect_data(
            self.train_episodes_per_cycle, top_level, trajectories_file=self._get_trajectories_file(global_step),
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

    def collect_data(self, count, top_level, trajectories_file=None, is_train=True, use_fixed_start_goal_pairs=False):
        print_and_log('collecting {} {} episodes of level {}'.format(count, 'train' if is_train else 'test', top_level))
        if use_fixed_start_goal_pairs:
            episode_results = self.episode_runner.play_fixed_episodes(top_level, is_train)
        else:
            episode_results = self.episode_runner.play_random_episodes(count, top_level, is_train)
        successes, accumulated_cost, dataset, endpoints_by_path = self._process_episode_results(
            episode_results, top_level)
        # write trajectory file
        if trajectories_file is not None:
            with open(trajectories_file, 'w') as f:
                for path_id in endpoints_by_path:
                    endpoints, is_valid_episode = endpoints_by_path[path_id]
                    f.write('path_id_{}{}'.format(path_id, os.linesep))
                    f.write('{}{}'.format('success' if is_valid_episode else 'collision', os.linesep))
                    for e in endpoints:
                        f.write('{}{}'.format(str(e), os.linesep))
                f.flush()
        print_and_log(
            'data collection done, success rate is {}, accumulated cost is {}'.format(successes, accumulated_cost))
        return successes, accumulated_cost, dataset

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

    def _get_trajectories_file(self, global_step):
        if global_step % self.steps_per_trajectory_print == 0:
            return os.path.join(self.train_trajectories_dir, '{}.txt'.format(global_step))
        else:
            return None

    def _get_relevant_levels(self, top_level):
        if self.train_levels == 'all-below':
            return range(1, top_level + 1)
        elif self.train_levels == 'topmost':
            return range(top_level, top_level + 1)
        else:
            assert False

    def print_gradient(self, count, level, cycle):
        if not self.check_gradients:
            return
        results = {}
        for i in range(count):
            s, g = self.episode_runner.game.get_free_random_state(), self.episode_runner.game.get_free_random_state()
            results[i] = self.get_gradient_print_info_single_start_goal(s, g, level)
        self.print_gradient_infos(results, cycle)

    def get_gradient_print_info_single_start_goal(self, start, goal, level):
        if not self.check_gradients:
            return
        self.gradient_saver.save(self.sess, 0)
        start_goal_pair = [(np.array(start), np.array(goal))]
        pre_train_mean = self._get_mean(start_goal_pair, level)
        middles, costs = self._take_single_step(start_goal_pair, level)
        post_train_mean = self._get_mean(start_goal_pair, level)
        self.gradient_saver.restore(self.sess)
        post_restore_mean = self._get_mean(start_goal_pair, level)
        assert all(np.equal(pre_train_mean, post_restore_mean))
        return start, goal, pre_train_mean, post_train_mean, middles

    def print_gradient_infos(self, gradient_results, cycle):
        gradient_output_file = os.path.join(self.gradient_output_dir, '{}.txt'.format(cycle))
        with open(gradient_output_file, 'w') as results_file:
            for result_id in gradient_results:
                results_file.write('id_{}{}'.format(result_id, os.linesep))
                start, goal, pre_train_mean, post_train_mean, middles = gradient_results[result_id]
                results_file.write('{}{}'.format(start, os.linesep))
                results_file.write('{}{}'.format(goal, os.linesep))
                results_file.write('{}{}'.format(pre_train_mean, os.linesep))
                results_file.write('{}{}'.format(post_train_mean, os.linesep))
                for middle in middles:
                    results_file.write('{}{}'.format(middle, os.linesep))
            results_file.flush()

    def _get_mean(self, start_goal_pair, level):
        test_episode_results = self.episode_runner.play_episodes(start_goal_pair, level, False)
        splits_top_level = test_episode_results[test_episode_results.keys()[0]][1][level]
        return splits_top_level[splits_top_level.keys()[0]][2]

    def _take_single_step(self, start_goal_pair, top_level):
        # run single prediction
        episode_results = self.episode_runner.play_episodes(start_goal_pair, top_level, True)
        # process the results
        _, _, dataset, _ = self._process_episode_results(episode_results, top_level)
        dataset = dataset[top_level]
        # modify the costs
        starts, ends, middles, _, _, costs = zip(*dataset)
        costs = self._process_costs(starts, ends, costs, top_level)
        # take a single policy step
        self.network.train_policy(top_level, starts, ends, middles, costs, self.sess)
        return middles, costs