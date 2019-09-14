import yaml
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

    config_copy['general']['training_cycles'] = 2000
    config_copy['general']['test_episodes'] = 100
    config_copy['general']['save_every_cycles'] = 20000  # don't save model
    config_copy['general']['cycles_per_trajectory_print'] = 20000  # don't save train trajectories

    config_copy['general']['train_episodes_per_cycle'] = random.choice([1000, 500, 100, 50])

    config_copy['cost']['collision_cost'] = random.choice([20., 100., 200., 1000.])
    config_copy['cost']['type'] = random.choice(['linear', 'huber'])

    config_copy['policy']['learning_rate'] = random.choice([0.001, 0.0001, 0.00001, 0.000001])
    config_copy['policy']['learning_rate_decrease_rate'] = random.choice([0.1, 0.5, 0.8, 1.0])

    layers_size = random.choice([20, 50, 100, 400])
    number_of_layers = random.choice([2, 3, 4])

    config_copy['policy']['layers'] = [layers_size] * number_of_layers
    config_copy['policy']['activation'] = random.choice(['relu', 'elu', 'tanh'])
    config_copy['policy']['base_std'] = random.choice([0.0, 0.05, 0.25, 0.5])
    config_copy['policy']['log_prob_ignore_pdf'] = random.choice([0.001, 0.00001, 0.0000001])
    config_copy['policy']['learn_std'] = random.choice([True, False])

    return config_copy


if __name__ == '__main__':
    run_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    tries = 5

    # read the config
    config = read_config()

    # set the path of the best model
    best_config_save_path = os.path.join(get_base_directory(), 'best_config')
    init_dir(best_config_save_path)
    best_config_save_path = os.path.join(best_config_save_path, 'best_config_{}.yml'.format(run_name))
    scores_save_path = os.path.join(best_config_save_path, 'scores_{}.yml'.format(run_name))

    best_score, best_config = None, None
    # while True:
    for _ in range(20):
        current_config = _modify_config(config)
        try:
            success_rates = []
            for _ in range(tries):
                tf.reset_default_graph()
                current_score = run_for_config(current_config)
                success_rates.append(current_score)
                if best_score is not None:
                    optimistic_scores = [1.] * (tries-len(success_rates))
                    optimistic_scores.extend(deepcopy(success_rates))
                    if np.mean(optimistic_scores) < best_score:
                        # no point to test this config further
                        break
            final_success_rate = np.mean(success_rates)
            if best_score is None or final_success_rate > best_score:
                copy_config(current_config, best_config_save_path)
                with open(scores_save_path, 'w') as scores_file:
                    scores_file.write('final score: {} all success rates: {}'.format(final_success_rate, success_rates))
                best_score = final_success_rate
                best_config = current_config
        except:
            print('#### error, run threw exception ####')
