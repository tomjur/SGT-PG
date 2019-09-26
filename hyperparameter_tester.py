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

    config_copy['general']['training_cycles'] = 10000
    config_copy['general']['train_episodes_per_cycle'] = random.choice([10, 50, 100, 500, 1000])

    config_copy['model']['reset_best_every'] = random.choice([0, 10, 100])
    config_copy['model']['repeat_train_trajectories'] = random.choice([10, 20, 100])

    lr_power = random.choice([2., 3., 4., 5.])
    lr_coeff = random.choice([1., 2.5, 5.])
    lr = float(lr_coeff * np.power(10, -lr_power))
    config_copy['policy']['learning_rate'] = lr
    config_copy['policy']['learning_rate_minimum'] = lr / 100.
    config_copy['policy']['learning_rate_decrease_rate'] = random.choice([1.0, 0.8, 0.5, 0.1])

    layers_size = random.choice([5, 10, 100])
    layers_number = random.choice([2, 3, 4, 5])
    config_copy['policy']['layers'] = [layers_size] * layers_number
    config_copy['policy']['base_std'] = random.choice([0.01, 0.05, 0.1, 0.5])

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
    failure_cost = 15.

    # read the config
    config = read_config()

    # set the files for logs etc.
    working_dir = os.path.join(get_base_directory(), 'best_config')
    init_dir(working_dir)
    best_config_save_path = os.path.join(working_dir, 'best_config_{}.yml'.format(run_name))
    general_log = os.path.join(working_dir, 'general_log_{}.txt'.format(run_name))

    # track the best score
    best_cost, best_config = None, None

    # while True:
    for group_index in range(number_of_groups):
        group_config = os.path.join(working_dir, 'group_{}_config_{}.yml'.format(group_index, run_name))
        group_log = os.path.join(working_dir, 'group_{}_log_{}.txt'.format(group_index, run_name))

        current_config = _modify_config(config)
        costs = []

        copy_config(current_config, group_config)
        final_cost = None
        try:
            for try_index in range(tries_per_config):
                tf.reset_default_graph()
                current_cost = run_for_config(current_config)
                write_to_text_log(group_log, 'try {} cost: {}'.format(try_index, current_cost))
                if current_cost > failure_cost:
                    # no point to test this config further
                    write_to_text_log(
                        group_log, 'aborting, found cost {} greater than failure cost threshold {}'.format(
                            current_cost, failure_cost))
                    break
                costs.append(current_cost)
                # check optimistic score
                if best_cost is not None:
                    optimistic_costs = [0.] * (tries_per_config - len(costs))
                    optimistic_costs.extend(deepcopy(costs))
                    mean_optimistic_cost = np.mean(optimistic_costs)
                    if mean_optimistic_cost > best_cost:
                        # no point to test this config further
                        write_to_text_log(
                            group_log, 'aborting, mean optimistic cost {} greater than best cost {}'.format(
                                mean_optimistic_cost, best_cost))
                        break
            final_cost = np.mean(costs)
            write_to_text_log(group_log, 'final cost: {}'.format(final_cost))
        except:
            write_to_text_log(group_log, 'error, run threw exception!')
            write_to_text_log(general_log, 'group {} crashed'.format(group_index))
            continue

        if final_cost is not None:
            # see if best
            if best_cost is None or final_cost < best_cost:
                write_to_text_log(group_log, 'final cost is the best')
                write_to_text_log(
                    general_log, 'group {} has the new best cost {}'.format(group_index, final_cost))
                best_cost = final_cost
                best_config = current_config
                copy_config(current_config, best_config_save_path)
