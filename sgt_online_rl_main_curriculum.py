import datetime
import random
import numpy as np
import tensorflow as tf
import os
import time


from config_utils import read_config, copy_config
from episode_runner import EpisodeRunner
from model_saver import ModelSaver
from network import Network
from summaries_collector import SummariesCollector
from docker_path_helper import get_base_directory, init_dir
from log_utils import init_log, print_and_log, close_log


def _get_game(config):
    if config['general']['is_point_robot']:
        from point_robot_game import PointRobotGame
        return PointRobotGame(config)
    else:
        from openrave_game import OpenraveGame
        return OpenraveGame(config)


def _get_tests(config):
    if config['general']['is_point_robot']:
        tests = [
            ((x1, y1), (x2, y2))
            for x1 in [0.9, -0.9] for y1 in [0.9, -0.9] for x2 in [0.9, -0.9] for y2 in [0.9, -0.9]
            if x1 != x2
        ]
        return tests
    else:
        return None


def _train_at_level(config, top_level, network, sess, dataset, global_step, summaries_collector):
    summaries_frequency = config['general']['write_summaries_every']
    batch_size = config['model']['batch_size']

    if config['model']['train_levels'] == 'all-below':
        levels_to_train = range(1, top_level+1)
    elif config['model']['train_levels'] == 'topmost':
        levels_to_train = range(top_level, top_level + 1)
    else:
        assert False
    for level in levels_to_train:
        valid_data = [
            (s, g, m, c) for (s, g, m, s_valid, g_valid, c) in dataset[level] if s_valid and g_valid
        ]
        if len(valid_data) == 0:
            print_and_log('############### no valid data')
            continue
        starts, ends, middles, costs = zip(*random.sample(valid_data, min(batch_size, len(valid_data))))
        summaries, prediction_loss, _ = network.train_policy(level, starts, ends, middles, costs, sess)
        if global_step % summaries_frequency == summaries_frequency - 1:
            summaries_collector.write_train_optimization_summaries(summaries, global_step)

        global_step += 1
    return global_step


def _collect_data(config, count, top_level, episode_runner, trajectories_dir=None, is_train=True):
    print_and_log('collecting {} {} episodes of level {}'.format(count, 'train' if is_train else 'test', top_level))
    episode_results = episode_runner.play_random_episodes(count, top_level, is_train)
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
        total_cost = splits[top_level][(0, len(endpoints)-1)][-1]
        accumulated_cost.append(total_cost)

        # extend the dataset
        gain = config['model']['gain']
        for level in range(1, top_level + 1):
            current_dataset = splits[level].values()
            if gain == 'full-traj':
                current_dataset = [
                    (s, g, m, s_valid, g_valid, total_cost) for (s, g, m, s_valid, g_valid, c) in current_dataset
                ]
            elif gain == 'future-only':
                pass
            else:
                assert False
            dataset[level].extend(current_dataset)

    successes = np.mean(successes)
    accumulated_cost = np.mean(accumulated_cost)
    print_and_log(
        'data collection done, success rate is {}, accumulated cost is {}'.format(successes, accumulated_cost))
    return successes, accumulated_cost, dataset


def run_for_config(config):
    # set the name of the model
    model_name = config['general']['name']
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    model_name = now + '_' + model_name if model_name is not None else now

    # where we save all the outputs
    scenario = config['general']['scenario']
    working_dir = os.path.join(get_base_directory(), scenario)
    init_dir(working_dir)

    saver_dir = os.path.join(working_dir, 'models', model_name)
    init_dir(saver_dir)
    init_log(log_file_path=os.path.join(saver_dir, 'log.txt'))
    copy_config(config, os.path.join(saver_dir, 'config.yml'))

    # generate graph:
    network = Network(config)

    # save model
    latest_saver = ModelSaver(os.path.join(saver_dir, 'latest_model'), 2, 'latest')
    best_saver = ModelSaver(os.path.join(saver_dir, 'best_model'), 1, 'best')

    summaries_collector = SummariesCollector(os.path.join(working_dir, 'tensorboard', model_name), model_name)

    with tf.Session(
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
            )
    ) as sess:
        sess.run(tf.global_variables_initializer())

        game = _get_game(config)
        policy_function = lambda starts, goals, level, is_train: network.predict_policy(
            starts, goals, level, sess, is_train)
        episode_runner = EpisodeRunner(config, game, policy_function)

        decrease_learn_rate_if_static_success = config['model']['decrease_learn_rate_if_static_success']
        stop_training_after_learn_rate_decrease = config['model']['stop_training_after_learn_rate_decrease']
        save_frequency = config['general']['save_every_cycles']
        train_episodes_per_cycle = config['general']['train_episodes_per_cycle']
        test_episodes = config['general']['test_episodes']
        test_frequency = config['general']['test_frequency']

        current_level = config['model']['starting_level']
        global_step = 0
        best_success_rate, best_success_global_step = None, None
        no_test_improvement, consecutive_learn_rate_decrease = 0, 0

        for cycle in range(config['general']['training_cycles']):
            if current_level > config['model']['levels']:
                print_and_log('trained all {} levels - needs to stop'.format(current_level))
                break
            print_and_log('starting cycle {}, level {}'.format(cycle, current_level))

            if cycle % config['general']['cycles_per_trajectory_print'] == 0:
                trajectories_dir = os.path.join(working_dir, 'trajectories', model_name, str(global_step))
            else:
                trajectories_dir = None
            successes, accumulated_cost, dataset = _collect_data(
                config, train_episodes_per_cycle, current_level, episode_runner, trajectories_dir, True)
            if global_step % config['general']['write_summaries_every']:
                summaries_collector.write_train_success_summaries(sess, global_step, successes)

            global_step = _train_at_level(
                config, current_level, network, sess, dataset, global_step, summaries_collector)
            print_and_log('done training cycle {} global step {}'.format(cycle, global_step))
            # save every now and then
            if cycle % save_frequency == save_frequency - 1:
                latest_saver.save(sess, global_step=global_step)

            if cycle % test_frequency == 0 or best_success_rate is None or best_success_rate < successes:
                # do test
                test_trajectories_dir = os.path.join(working_dir, 'test_trajectories', model_name, str(global_step))
                test_successes, test_cost, _ = _collect_data(
                    config, test_episodes, current_level, episode_runner, test_trajectories_dir, False)
                summaries_collector.write_test_success_summaries(sess, global_step, test_successes)
                # decide how to act next
                print_and_log('old success rate was {} at step {}'.format(best_success_rate, best_success_global_step))

                should_increase_level = False
                print_and_log('current learn rates {}'.format(network.get_learn_rates(sess, current_level)))
                if best_success_rate is None or test_successes > best_success_rate:
                    print_and_log('new best success rate {} at step {}'.format(test_successes, global_step))
                    best_success_rate, best_success_global_step = test_successes, global_step
                    no_test_improvement = 0
                    consecutive_learn_rate_decrease = 0
                    best_saver.save(sess, global_step)

                    # if test_successes == 1.:
                    #     should_increase_level = True
                else:
                    print_and_log('new model is not the best')
                    no_test_improvement += 1
                    print_and_log('no improvement count {} of {}'.format(
                        no_test_improvement, decrease_learn_rate_if_static_success))
                    if no_test_improvement == decrease_learn_rate_if_static_success:
                        if config['model']['train_levels'] == 'all-below':
                            levels_to_decrease_learn_rate = range(1, current_level + 1)
                        elif config['model']['train_levels'] == 'topmost':
                            levels_to_decrease_learn_rate = range(current_level, current_level + 1)
                        else:
                            assert False
                        for l in levels_to_decrease_learn_rate:
                            network.decrease_learn_rates(sess, l)
                        no_test_improvement = 0
                        consecutive_learn_rate_decrease += 1
                        print_and_log('decreasing learn rates {} of {}'.format(
                            consecutive_learn_rate_decrease, stop_training_after_learn_rate_decrease)
                        )
                        print_and_log('new learn rates {}'.format(network.get_learn_rates(sess, current_level)))
                        if consecutive_learn_rate_decrease == stop_training_after_learn_rate_decrease:
                            should_increase_level = True

                if should_increase_level:
                    best_saver.restore(sess)
                    current_level += 1
                    if current_level <= config['model']['levels']:
                        best_success_rate, best_success_global_step = None, None
                        no_test_improvement, consecutive_learn_rate_decrease = 0, 0
                        print_and_log('initiating level {} from previous level'.format(current_level))
                        if config['model']['init_from_lower_level']:
                            network.init_policy_from_lower_level(sess, current_level)

            # mark in log the end of cycle
            print_and_log(os.linesep)

        print_and_log('end of run best: {} from step: {}'.format(best_success_rate, best_success_global_step))
        close_log()
        return best_success_rate


if __name__ == '__main__':
    # read the config
    config = read_config()
    run_for_config(config)
