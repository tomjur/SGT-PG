import random
import numpy as np

from log_utils import print_and_log


class TrainerSequential:
    def __init__(self,  model_name, config, working_dir, network, sess, episode_runner, summaries_collector):
        self.model_name = model_name
        self.config = config
        self.working_dir = working_dir
        self.network = network
        self.sess = sess
        self.episode_runner = episode_runner
        self.summaries_collector = summaries_collector

        self.batch_size = config['model']['batch_size']
        self.steps_per_trajectory_print = config['general']['cycles_per_trajectory_print']
        self.train_episodes_per_cycle = config['general']['train_episodes_per_cycle']
        self.gamma = config['model']['gamma']

    def train_policy(self, global_step):
        successes, accumulated_cost, dataset, _ = self.collect_data(
            self.train_episodes_per_cycle, is_train=True, use_fixed_start_goal_pairs=False)
        self.summaries_collector.write_train_success_summaries(self.sess, global_step, successes, accumulated_cost)

        # set the baseline to the current policy
        self.network.update_baseline_policy(self.sess)
        # do optimization steps
        for update_step in range(self.config['model']['consecutive_optimization_steps']):
            states, goals, actions, costs = zip(*random.sample(dataset, min(self.batch_size, len(dataset))))
            costs = np.expand_dims(np.array(costs), axis=-1)
            summaries, prediction_loss, _ = self.network.train_policy(states, goals, actions, costs, self.sess)
            self.summaries_collector.write_train_optimization_summaries(summaries, global_step)
            global_step += 1
        return global_step

    def collect_data(self, count, is_train=True, use_fixed_start_goal_pairs=False):
        print_and_log('collecting {} {} episodes'.format(count, 'train' if is_train else 'test'))
        if use_fixed_start_goal_pairs:
            episode_results = self.episode_runner.play_fixed_episodes(is_train)
        else:
            episode_results = self.episode_runner.play_random_episodes(count, is_train)
        successes, accumulated_cost, dataset, endpoints_by_path = self._process_episode_results(episode_results)
        print_and_log(
            'data collection done, success rate is {}, accumulated cost is {}'.format(successes, accumulated_cost))
        return successes, accumulated_cost, dataset, endpoints_by_path

    def _process_episode_results(self, episode_results):
        accumulated_cost, successes = [], []
        dataset = []
        endpoints_by_path = {}

        for path_id in episode_results:
            states, goal, actions, costs, is_successful = episode_results[path_id]
            assert len(states) == len(actions)+1
            assert len(costs) == len(actions)

            endpoints_by_path[path_id] = (states, actions, is_successful)  # data used to visualize the episode

            # log collision
            successes.append(is_successful)

            # compute costs
            discounted_future_costs = list(reversed(costs[:]))
            for i in range(len(costs)-1):
                discounted_future_costs[i+1] += self.gamma * discounted_future_costs[i]
            discounted_future_costs = list(reversed(discounted_future_costs))
            accumulated_cost.append(discounted_future_costs[0])
            assert len(costs) == len(discounted_future_costs)

            # extend the dataset
            for i in range(len(actions)):
                transition = (states[i], goal, actions[i], discounted_future_costs[i])
                dataset.append(transition)

        successes = np.mean(successes)
        accumulated_cost = np.mean(accumulated_cost)
        return successes, accumulated_cost, dataset, endpoints_by_path