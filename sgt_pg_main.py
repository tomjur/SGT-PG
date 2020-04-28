import datetime
import tensorflow as tf
import numpy as np
import os
import time


from config_utils import read_config, copy_config
from episode_runner_subgoal import EpisodeRunnerSubgoal
from model_saver import ModelSaver
from network_subgoal import Network
from summaries_collector import SummariesCollector
from path_helper import get_base_directory, init_dir, serialize_compress
from log_utils import init_log, print_and_log, close_log
from trainer_subgoal import TrainerSubgoal


def _get_game(config):
    scenario = config['general']['scenario']
    if 'point_robot' in scenario:
        from point_robot_game_subgoal import PointRobotGameSubgoal
        return PointRobotGameSubgoal(scenario)
    elif 'panda' in scenario:
        from panda_game_subgoal import PandaGameSubgoal
        max_cores = config['general']['train_episodes_per_cycle'] * config['model']['repeat_train_trajectories']
        limit_workers = config['general']['limit_workers']
        if limit_workers is not None:
            max_cores = min(limit_workers, max_cores)
        return PandaGameSubgoal(scenario, max_cores=max_cores)
    elif 'disks' in scenario:
        from disks_subgoal_game import DisksSubgoalGame
        shaping_coef = float(scenario.split('_')[1])
        return DisksSubgoalGame(shaping_coeff=shaping_coef)
    else:
        assert False


def get_initial_curriculum(config):
    if config['curriculum']['use']:
        return config['policy']['base_std'] * config['curriculum']['times_std_start_coefficient']
    else:
        return None


def run_for_config(config):
    # set the name of the model
    model_name = config['general']['name']
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    model_name = now + '_' + model_name if model_name is not None else now

    # where we save all the outputs
    scenario = config['general']['scenario']
    working_dir = os.path.join(get_base_directory(), 'sgt', scenario)
    init_dir(working_dir)

    saver_dir = os.path.join(working_dir, 'models', model_name)
    init_dir(saver_dir)
    init_log(log_file_path=os.path.join(saver_dir, 'log.txt'))
    copy_config(config, os.path.join(saver_dir, 'config.yml'))
    episodic_success_rates_path = os.path.join(saver_dir, 'results.txt')
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

        def policy_function(starts, goals, level, is_train):
            res = network.predict_policy(starts, goals, level, sess, is_train)
            means = 0.5 * (np.array(starts) + np.array(goals))
            distance = np.linalg.norm(res[0] - means, axis=1)
            print(f'distance from mean: mean {distance.mean()} min {distance.min()} max {distance.max()}')
            if np.any(np.isnan(res)):
                print_and_log('######################## Nan predictions detected...')
            return res

        episode_runner = EpisodeRunnerSubgoal(config, game, policy_function)
        trainer = TrainerSubgoal(model_name, config, working_dir, network, sess, episode_runner, summaries_collector,
                                 curriculum_coefficient=get_initial_curriculum(config))

        decrease_learn_rate_if_static_success = config['model']['decrease_learn_rate_if_static_success']
        stop_training_after_learn_rate_decrease = config['model']['stop_training_after_learn_rate_decrease']
        reset_best_every = config['model']['reset_best_every']

        global_step = 0
        best_curriculum_coefficient = None

        for current_level in range(config['model']['starting_level'], config['model']['levels']+1):

            best_cost, best_cost_global_step = None, None
            no_test_improvement, consecutive_learn_rate_decrease = 0, 0

            if config['model']['init_from_lower_level'] and current_level > 1:
                print_and_log('initiating level {} from previous level'.format(current_level))
                network.init_policy_from_lower_level(sess, current_level)

            for cycle in range(config['general']['training_cycles_per_level']):
                print_and_log('starting cycle {}, level {}'.format(cycle, current_level))

                new_global_step, success_ratio = trainer.train_policy_at_level(current_level, global_step)
                if new_global_step == global_step:
                    print_and_log('no data found in training cycle {} global step still {}'.format(cycle, global_step))
                    continue
                else:
                    global_step = new_global_step

                if (cycle + 1) % config['policy']['decrease_std_every'] == 0:
                    network.decrease_base_std(sess, current_level)
                    print_and_log('new base stds {}'.format(network.get_base_stds(sess, current_level)))

                print_and_log('done training cycle {} global step {}'.format(cycle, global_step))

                # save every now and then
                if cycle % config['general']['save_every_cycles'] == 0:
                    latest_saver.save(sess, global_step=global_step)

                if cycle % config['general']['test_frequency'] == 0:
                    # do test
                    test_successes, test_cost, _, endpoints_by_path = trainer.collect_test_data(current_level, False)
                    summaries_collector.write_test_success_summaries(
                        sess, global_step, test_successes, test_cost, trainer.curriculum_coefficient)
                    with open(episodic_success_rates_path, 'a') as f:
                        f.write('{} {} {} {} {}'.format(
                            current_level, trainer.train_episodes_counter, test_successes, test_cost, os.linesep))

                    # decide how to act next
                    print_and_log('old cost was {} at step {}'.format(best_cost, best_cost_global_step))
                    print_and_log('current learn rates {}'.format(network.get_learn_rates(sess, current_level)))
                    print_and_log('current base stds {}'.format(network.get_base_stds(sess, current_level)))
                    if best_cost is None or test_cost < best_cost:
                        print_and_log('new best cost {} at step {}'.format(test_cost, global_step))
                        best_cost, best_cost_global_step = test_cost, global_step
                        best_curriculum_coefficient = trainer.curriculum_coefficient
                        no_test_improvement, consecutive_learn_rate_decrease = 0, 0
                        best_saver.save(sess, global_step)
                        test_trajectories_file = os.path.join(test_trajectories_dir, '{}.txt'.format(global_step))
                        serialize_compress(endpoints_by_path, test_trajectories_file)
                    else:
                        print_and_log('new model is not the best with cost {} at step {}'.format(
                            test_cost, global_step))
                        no_test_improvement += 1
                        print_and_log('no improvement count {} of {}'.format(
                            no_test_improvement, decrease_learn_rate_if_static_success))
                        if reset_best_every > 0 and no_test_improvement % reset_best_every == reset_best_every - 1:
                            # restore the model every once in a while if did not find a better solution in a while
                            restore_best(sess, best_saver, best_curriculum_coefficient, trainer)
                        if no_test_improvement == decrease_learn_rate_if_static_success:
                            # restore the best model
                            if config['model']['restore_on_decrease']:
                                restore_best(sess, best_saver, best_curriculum_coefficient, trainer)
                            # decrease learn rates
                            network.decrease_learn_rates(sess, current_level)
                            no_test_improvement = 0
                            consecutive_learn_rate_decrease += 1
                            print_and_log('decreasing learn rates {} of {}'.format(
                                consecutive_learn_rate_decrease, stop_training_after_learn_rate_decrease)
                            )
                            print_and_log('new learn rates {}'.format(network.get_learn_rates(sess, current_level)))
                            if consecutive_learn_rate_decrease == stop_training_after_learn_rate_decrease:
                                break

                if trainer.curriculum_coefficient is not None:
                    if success_ratio > config['curriculum']['raise_when_train_above']:
                        print_and_log('current curriculum coefficient {}'.format(trainer.curriculum_coefficient))
                        trainer.curriculum_coefficient *= config['curriculum']['raise_times']
                        print_and_log('curriculum coefficient raised to {}'.format(trainer.curriculum_coefficient))

                # mark in log the end of cycle
                print_and_log(os.linesep)

            # if we finished because we ran out of cycles, we still need to make one more test
            end_of_level_test(best_cost, best_cost_global_step, best_curriculum_coefficient, best_saver, sess,
                              test_trajectories_dir, trainer, current_level)

        print_and_log('trained all levels - needs to stop')
        close_log()
        return best_cost


def end_of_level_test(best_cost, best_cost_global_step, best_curriculum_coefficient, best_saver, sess,
                      test_trajectories_dir, trainer, level):
    print_and_log('end of level {} best: {} from step: {}'.format(level, best_cost, best_cost_global_step))
    restore_best(sess, best_saver, best_curriculum_coefficient, trainer)
    # test all
    test_trajectories_file = os.path.join(test_trajectories_dir, 'level{}_all.txt'.format(level))
    endpoints_by_path = trainer.collect_test_data(level, is_challenging=False)[-1]
    serialize_compress(endpoints_by_path, test_trajectories_file)
    print_and_log(os.linesep)
    # test hard
    test_trajectories_file = os.path.join(test_trajectories_dir, 'level{}_challenging.txt'.format(level))
    endpoints_by_path = trainer.collect_test_data(level, is_challenging=True)[-1]
    serialize_compress(endpoints_by_path, test_trajectories_file)
    print_and_log(os.linesep)


def restore_best(sess, best_saver, best_curriculum_coefficient, trainer):
    best_saver.restore(sess)
    trainer.episode_runner.curriculum_coefficient = best_curriculum_coefficient


if __name__ == '__main__':
    # read the config
    config = read_config()
    run_for_config(config)
