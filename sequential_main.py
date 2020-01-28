import datetime
import tensorflow as tf
import os
import time


from config_utils import read_config, copy_config
from episode_runner_sequential import EpisodeRunnerSequential
from model_saver import ModelSaver
from network_sequential import NetworkSequential
from summaries_collector import SummariesCollector
from path_helper import get_base_directory, init_dir, serialize_compress, get_config_directory
from log_utils import init_log, print_and_log, close_log
from trainer_sequential import TrainerSequential


def _get_game(config):
    scenario = config['general']['scenario']
    if 'point_robot' in scenario:
        from point_robot_game_sequential import PointRobotGameSequential
        return PointRobotGameSequential(scenario, config['cost']['collision_cost'], config['cost']['goal_reward'],
                                        max_action_limit=config['cost']['goal_closeness_distance'],
                                        max_steps=config['model']['max_steps'],
                                        goal_closeness_distance=config['cost']['goal_closeness_distance'])
    if 'panda' in scenario:
        from panda_game_sequential import PandaGameSequential
        max_cores = max(config['general']['test_episodes'], config['general']['train_episodes_per_cycle'])
        limit_workers = config['panda_game']['limit_workers']
        if limit_workers is not None:
            max_cores = min(limit_workers, max_cores)
        return PandaGameSequential(config, max_cores)
    else:
        assert False


def get_initial_curriculum(config):
    if config['curriculum']['use']:
        return config['curriculum']['times_goal_start_coefficient']
    else:
        return None


def run_for_config(config):
    # set the name of the model
    model_name = config['general']['name']
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    model_name = now + '_' + model_name if model_name is not None else now

    # where we save all the outputs
    scenario = config['general']['scenario']
    working_dir = os.path.join(get_base_directory(), 'sequential', scenario)
    init_dir(working_dir)

    saver_dir = os.path.join(working_dir, 'models', model_name)
    init_dir(saver_dir)
    init_log(log_file_path=os.path.join(saver_dir, 'log.txt'))
    copy_config(config, os.path.join(saver_dir, 'config.yml'))
    episodic_success_rates_path = os.path.join(saver_dir, 'results.txt')
    weights_log_dir = os.path.join(saver_dir, 'weights_logs')
    init_dir(weights_log_dir)
    test_trajectories_dir = os.path.join(working_dir, 'test_trajectories', model_name)
    init_dir(test_trajectories_dir)

    # generate game
    game = _get_game(config)

    network = NetworkSequential(config, game.get_state_space_size(), game.get_action_space_size(), is_rollout_agent=False)
    network_variables = network.get_all_variables()

    # save model
    latest_saver = ModelSaver(os.path.join(saver_dir, 'latest_model'), 2, 'latest', variables=network_variables)
    best_saver = ModelSaver(os.path.join(saver_dir, 'best_model'), 1, 'best', variables=network_variables)

    summaries_collector = SummariesCollector(os.path.join(working_dir, 'tensorboard', model_name), model_name)

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
    )) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        episode_runner = EpisodeRunnerSequential(config, game, curriculum_coefficient=get_initial_curriculum(config))

        trainer = TrainerSequential(model_name, config, working_dir, network, sess, episode_runner, summaries_collector)

        decrease_learn_rate_if_static_success = config['model']['decrease_learn_rate_if_static_success']
        stop_training_after_learn_rate_decrease = config['model']['stop_training_after_learn_rate_decrease']
        reset_best_every = config['model']['reset_best_every']

        global_step = 0
        best_cost, best_cost_global_step, best_curriculum_coefficient = None, None, None
        no_test_improvement, consecutive_learn_rate_decrease = 0, 0

        for cycle in range(config['general']['training_cycles']):
            print_and_log('starting cycle {}'.format(cycle))

            global_step, success_ratio = trainer.train_policy(global_step)

            if (cycle+1) % config['policy']['decrease_std_every'] == 0:
                network.decrease_base_std(sess)
                print_and_log('new base stds {}'.format(network.get_base_std(sess)))

            print_and_log('done training cycle {} global step {}'.format(cycle, global_step))

            # save every now and then
            if cycle % config['general']['save_every_cycles'] == 0:
                latest_saver.save(sess, global_step=global_step)

            if cycle % config['general']['test_frequency'] == 0:
                # do test
                test_successes, test_cost, _, endpoints_by_path = trainer.collect_data(
                    config['general']['test_episodes'], is_train=False, use_fixed_start_goal_pairs=True)
                summaries_collector.write_test_success_summaries(
                    sess, global_step, test_successes, test_cost, episode_runner.curriculum_coefficient
                )
                with open(episodic_success_rates_path, 'a') as f:
                    f.write('{} {} {} {}'.format(trainer.train_episodes_counter, test_successes, test_cost, os.linesep))

                # decide how to act next
                print_and_log('old cost was {} at step {}'.format(best_cost, best_cost_global_step))
                print_and_log('current learn rates {}'.format(network.get_learn_rate(sess)))
                print_and_log('current base stds {}'.format(network.get_base_std(sess)))
                if best_cost is None or test_cost < best_cost:
                    print_and_log('new best cost {} at step {}'.format(test_cost, global_step))
                    best_cost, best_cost_global_step = test_cost, global_step
                    best_curriculum_coefficient = episode_runner.curriculum_coefficient
                    no_test_improvement = 0
                    consecutive_learn_rate_decrease = 0
                    best_saver.save(sess, global_step)
                    test_trajectories_file = os.path.join(test_trajectories_dir, '{}.txt'.format(global_step))
                    serialize_compress(endpoints_by_path, test_trajectories_file)
                else:
                    print_and_log('new model is not the best with cost {} at step {}'.format(test_cost, global_step))
                    no_test_improvement += 1
                    print_and_log('no improvement count {} of {}'.format(
                        no_test_improvement, decrease_learn_rate_if_static_success))
                    if reset_best_every > 0 and no_test_improvement % reset_best_every == reset_best_every - 1:
                        # restore the model every once in a while if did not find a better solution in a while
                        best_saver.restore(sess)
                        episode_runner.curriculum_coefficient = best_curriculum_coefficient
                    if no_test_improvement == decrease_learn_rate_if_static_success:
                        # restore the best model
                        if config['model']['restore_on_decrease']:
                            best_saver.restore(sess)
                            episode_runner.curriculum_coefficient = best_curriculum_coefficient
                        network.decrease_learn_rates(sess)

                        no_test_improvement = 0
                        consecutive_learn_rate_decrease += 1
                        print_and_log('decreasing learn rates {} of {}'.format(
                            consecutive_learn_rate_decrease, stop_training_after_learn_rate_decrease)
                        )
                        print_and_log('new learn rates {}'.format(network.get_learn_rate(sess)))
                        if consecutive_learn_rate_decrease == stop_training_after_learn_rate_decrease:
                            print_and_log('needs to stop')
                            best_saver.restore(sess)
                            break

            if episode_runner.curriculum_coefficient is not None:
                if success_ratio > config['curriculum']['raise_when_train_above']:
                    print_and_log('current curriculum coefficient {}'.format(episode_runner.curriculum_coefficient))
                    episode_runner.curriculum_coefficient *= config['curriculum']['raise_times']
                    print_and_log('curriculum coefficient raised to {}'.format(episode_runner.curriculum_coefficient))

            # mark in log the end of cycle
            print_and_log(os.linesep)

        print_and_log('end of run best: {} from step: {}'.format(best_cost, best_cost_global_step))
        print_and_log('testing on a new set of start-goal pairs')
        best_saver.restore(sess)
        test_trajectories_file = os.path.join(test_trajectories_dir, '-1.txt')
        endpoints_by_path = trainer.collect_data(
            config['general']['test_episodes'], is_train=False, use_fixed_start_goal_pairs=True
        )[-1]
        serialize_compress(endpoints_by_path, test_trajectories_file)

        close_log()
        return best_cost


if __name__ == '__main__':
    # read the config
    config_path = os.path.join(get_config_directory(), 'config_sequential.yml')
    config = read_config(config_path)
    run_for_config(config)
