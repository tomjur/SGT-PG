import datetime
import tensorflow as tf
import os
import time


from config_utils import read_config, copy_config
from episode_runner import EpisodeRunner
from model_saver import ModelSaver
from network import Network
from summaries_collector import SummariesCollector
from path_helper import get_base_directory, init_dir, serialize_compress
from log_utils import init_log, print_and_log, close_log
from trainer import Trainer


def _get_game(config):
    if 'point_robot' in config['general']['scenario']:
        from point_robot_game import PointRobotGame
        return PointRobotGame(config)
    if 'panda' in config['general']['scenario']:
        from panda_game import PandaGame
        return PandaGame(config)
    else:
        assert False


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


def print_policy_weights(sess, network, global_step, reason, logs_dir):
    names = [v.name for v in network.get_all_variables()]
    vars = sess.run(network.get_all_variables())
    weights_log_file = os.path.join(logs_dir, '{}_{}.txt'.format(global_step, reason))
    with open(weights_log_file, 'a') as f:
        for i in range(len(names)):
            f.write('var: {}{}'.format(names[i], os.linesep))
            f.write('{}{}'.format(vars[i], os.linesep))
        f.flush()


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
    weights_log_dir = os.path.join(saver_dir, 'weights_logs')
    init_dir(weights_log_dir)
    test_trajectories_dir = os.path.join(working_dir, 'test_trajectories', model_name)
    init_dir(test_trajectories_dir)

    # generate game
    game = _get_game(config)

    network = Network(config, game)
    network_variables = network.get_all_variables()

    # save model
    latest_saver = ModelSaver(os.path.join(saver_dir, 'latest_model'), 2, 'latest', variables=network_variables)
    best_saver = ModelSaver(os.path.join(saver_dir, 'best_model'), 1, 'best', variables=network_variables)

    summaries_collector = SummariesCollector(os.path.join(working_dir, 'tensorboard', model_name), model_name)

    with tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
            )
    ) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        policy_function = lambda starts, goals, level, is_train: network.predict_policy(
            starts, goals, level, sess, is_train)
        episode_runner = EpisodeRunner(config, game, policy_function)
        trainer = Trainer(model_name, config, working_dir, network, sess, episode_runner, summaries_collector)

        decrease_learn_rate_if_static_success = config['model']['decrease_learn_rate_if_static_success']
        stop_training_after_learn_rate_decrease = config['model']['stop_training_after_learn_rate_decrease']
        reset_best_every = config['model']['reset_best_every']

        current_level = config['model']['starting_level']
        global_step, first_global_step_of_level = 0, 0
        best_cost, best_cost_global_step = None, None
        no_test_improvement, consecutive_learn_rate_decrease = 0, 0

        if config['general']['weight_printing_frequency'] > 0:
            print_policy_weights(sess, network, global_step, 'init', weights_log_dir)

        for cycle in range(config['general']['training_cycles']):
            print_and_log('starting cycle {}, level {}'.format(cycle, current_level))

            new_global_step = trainer.train_policy_at_level(current_level, global_step)
            if new_global_step == global_step:
                print_and_log('no data found in training cycle {} global step still {}'.format(cycle, global_step))
                continue
            else:
                global_step = new_global_step

            if (global_step - first_global_step_of_level) % config['policy']['decrease_std_every'] == 0:
                network.decrease_base_std(sess, current_level)
                print_and_log('new base stds {}'.format(network.get_base_stds(sess, current_level)))

            print_and_log('done training cycle {} global step {}'.format(cycle, global_step))

            # save every now and then
            if cycle % config['general']['save_every_cycles'] == 0:
                latest_saver.save(sess, global_step=global_step)

            if cycle % config['general']['test_frequency'] == 0:
                # if trainer.check_gradients:
                    # trainer.print_gradient(
                    #     config['gradient_checker']['gradient_points_to_sample'], current_level, cycle)

                # do test
                test_successes, test_cost, _, endpoints_by_path = trainer.collect_data(
                    config['general']['test_episodes'], current_level, is_train=False, use_fixed_start_goal_pairs=True)
                summaries_collector.write_test_success_summaries(sess, global_step, test_successes, test_cost)

                # decide how to act next
                print_and_log('old cost was {} at step {}'.format(best_cost, best_cost_global_step))
                should_increase_level = False
                print_and_log('current learn rates {}'.format(network.get_learn_rates(sess, current_level)))
                print_and_log('current base stds {}'.format(network.get_base_stds(sess, current_level)))
                if best_cost is None or test_cost < best_cost:
                    print_and_log('new best cost {} at step {}'.format(test_cost, global_step))
                    best_cost, best_cost_global_step = test_cost, global_step
                    no_test_improvement = 0
                    consecutive_learn_rate_decrease = 0
                    best_saver.save(sess, global_step)
                    test_trajectories_file = os.path.join(test_trajectories_dir, '{}.txt'.format(global_step))
                    serialize_compress(endpoints_by_path, test_trajectories_file)
                    if config['general']['weight_printing_frequency'] > 0:
                        print_policy_weights(sess, network, global_step, 'best', weights_log_dir)
                else:
                    print_and_log('new model is not the best with cost {} at step {}'.format(test_cost, global_step))
                    no_test_improvement += 1
                    print_and_log('no improvement count {} of {}'.format(
                        no_test_improvement, decrease_learn_rate_if_static_success))
                    if reset_best_every > 0 and no_test_improvement % reset_best_every == reset_best_every - 1:
                        # restore the model every once in a while if did not find a better solution in a while
                        best_saver.restore(sess)
                    if no_test_improvement == decrease_learn_rate_if_static_success:
                        # restore the best model
                        if config['model']['restore_on_decrease']:
                            best_saver.restore(sess)
                        # decrease learn rates
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
                        first_global_step_of_level = global_step
                        best_cost, best_cost_global_step = None, None
                        no_test_improvement, consecutive_learn_rate_decrease = 0, 0
                        if config['model']['init_from_lower_level']:
                            print_and_log('initiating level {} from previous level'.format(current_level))
                            network.init_policy_from_lower_level(sess, current_level)
                    else:
                        # return the current level to its correct level for final prediction
                        current_level -= 1
                        print_and_log('trained all {} levels - needs to stop'.format(current_level))
                        break

            if config['general']['weight_printing_frequency'] > 0:
                if (cycle+1) % config['general']['weight_printing_frequency'] == 0:
                    print_policy_weights(sess, network, global_step, 'regular', weights_log_dir)

            # mark in log the end of cycle
            print_and_log(os.linesep)

        print_and_log('end of run best: {} from step: {}'.format(best_cost, best_cost_global_step))
        print_and_log('testing on a new set of start-goal pairs')
        test_trajectories_file = os.path.join(test_trajectories_dir, '-1.txt')
        endpoints_by_path = trainer.collect_data(
            config['general']['test_episodes'], current_level, is_train=False,
            use_fixed_start_goal_pairs=False
        )[-1]
        serialize_compress(endpoints_by_path, test_trajectories_file)
        if config['general']['weight_printing_frequency'] > 0:
            print_policy_weights(sess, network, global_step, 'final', weights_log_dir)

        close_log()
        return best_cost


if __name__ == '__main__':
    # read the config
    config = read_config()
    run_for_config(config)
