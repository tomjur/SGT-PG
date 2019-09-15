import datetime
import os
import time
import numpy as np
import tensorflow as tf
import random
from copy import deepcopy
from config_utils import read_config, copy_config
from docker_path_helper import init_dir, get_base_directory
from sgt_online_rl_main_curriculum import run_for_config


def _modify_config(config):
    config_copy = deepcopy(config)

    # config_copy['general']['training_cycles'] = 10
    config_copy['general']['training_cycles'] = 2000
    config_copy['general']['test_episodes'] = 100
    config_copy['general']['save_every_cycles'] = 20000  # don't save model
    config_copy['general']['cycles_per_trajectory_print'] = 20000  # don't save train trajectories

    config_copy['general']['train_episodes_per_cycle'] = random.choice([1000, 500, 100, 50])

    config_copy['cost']['collision_cost'] = random.choice([20., 100., 200., 1000.])
    config_copy['cost']['type'] = random.choice(['huber'])
    # config_copy['cost']['type'] = random.choice(['linear', 'huber'])

    config_copy['policy']['learning_rate'] = random.choice([0.001, 0.0001, 0.00001, 0.000001])
    config_copy['policy']['learning_rate_decrease_rate'] = random.choice([0.1, 0.5, 0.8, 1.0])

    layers_size = random.choice([5, 20, 50, 100, 400])
    number_of_layers = random.choice([2, 3, 4])

    config_copy['policy']['layers'] = [layers_size] * number_of_layers
    config_copy['policy']['activation'] = random.choice(['relu', 'elu', 'tanh'])
    config_copy['policy']['base_std'] = random.choice([0.0, 0.05, 0.25, 0.5])
    config_copy['policy']['log_prob_ignore_pdf'] = random.choice([0.001, 0.00001, 0.0000001])
    config_copy['policy']['learn_std'] = random.choice([True, False])

    return config_copy


def write_to_text_log(log_filepath, message):
    with open(log_filepath, 'a') as log_file:
        log_file.write(message + os.linesep)
        log_file.flush()


if __name__ == '__main__':
    run_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    tries_per_config = 5
    # tries_per_config = 2
    number_of_groups = 20
    # number_of_groups = 4

    # read the config
    config = read_config()

    # set the files for logs etc.
    working_dir = os.path.join(get_base_directory(), 'best_config')
    init_dir(working_dir)
    best_config_save_path = os.path.join(working_dir, 'best_config_{}.yml'.format(run_name))
    general_log = os.path.join(working_dir, 'general_log_{}.txt'.format(run_name))

    # track the best score
    best_score, best_config = None, None

    # while True:
    for group_index in range(number_of_groups):
        group_config = os.path.join(working_dir, 'group_{}_config_{}.yml'.format(group_index, run_name))
        group_log = os.path.join(working_dir, 'group_{}_log_{}.txt'.format(group_index, run_name))

        current_config = _modify_config(config)
        success_rates = []

        copy_config(current_config, group_config)
        final_success_rate = None
        try:
            for try_index in range(tries_per_config):
                tf.reset_default_graph()
                current_score = run_for_config(current_config)
                write_to_text_log(group_log, 'try {} score: {}'.format(try_index, current_score))
                success_rates.append(current_score)
                if best_score is not None:
                    optimistic_scores = [1.] * (tries_per_config - len(success_rates))
                    optimistic_scores.extend(deepcopy(success_rates))
                    mean_optimistic_score = np.mean(optimistic_scores)
                    if mean_optimistic_score < best_score:
                        write_to_text_log(
                            group_log, 'aborting, mean optimistic score {} smaller than best score {}'.format(
                                mean_optimistic_score, best_score))
                        # no point to test this config further
                        break
            final_success_rate = np.mean(success_rates)
            write_to_text_log(group_log, 'final score: {}'.format(final_success_rate))
        except:
            write_to_text_log(group_log, 'error, run threw exception!')
            write_to_text_log(general_log, 'group {} crashed'.format(group_index))
            continue

        if final_success_rate is not None:
            # see if best
            if best_score is None or final_success_rate > best_score:
                write_to_text_log(group_log, 'final score is the best')
                write_to_text_log(
                    general_log, 'group {} has the new best score {}'.format(group_index, final_success_rate))
                best_score = final_success_rate
                best_config = current_config
                copy_config(current_config, best_config_save_path)
