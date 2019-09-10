import os
import tensorflow as tf
import numpy as np

from config_utils import read_config
from docker_path_helper import get_base_directory
from model_saver import ModelSaver
from network import Network

def parse_trajectory_line(line):
    line = line.replace('[', '').replace(']', '')
    parts = line.split(', ')
    assert len(parts) == 2

    x1 = float(parts[0])
    y1 = float(parts[1])
    return x1, y1

if __name__ == '__main__':
    model_name = '2019_08_26_14_59_22'
    saver_global_step = '153000'
    trajectory_global_step = '128600'
    trajectory_name = 'success_310.txt'


    # read the config
    config = read_config()

    # where we save all the outputs
    scenario = config['general']['scenario']
    working_dir = os.path.join(get_base_directory(), scenario)

    saver_dir = os.path.join(working_dir, 'models', model_name)
    best_saver_path = os.path.join(saver_dir, 'best_model')

    # generate graph:
    network = Network(config, )
    best_saver = ModelSaver(best_saver_path, 1, 'best')

    # read trajectory
    trajectory_file_path = os.path.join(
        working_dir, 'trajectories', model_name, trajectory_global_step, trajectory_name)
    with open(trajectory_file_path, 'r') as f:
        endpoints = [parse_trajectory_line(l) for l in f.readlines()]
    start = endpoints[0]
    goal = endpoints[-1]
    mid = endpoints[(len(endpoints)-1) / 2]

    with tf.Session(
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
            )
    ) as sess:
        x = np.linspace(-1, 1, 500)
        y = np.linspace(-1, 1, 500)
        xv, yv = np.meshgrid(x, y)

        points = [(xv[i, j], yv[i, j]) for i in range(len(x)) for j in range(len(y))]

        restore_from = os.path.join(best_saver_path, 'best-{}'.format(saver_global_step))
        best_saver.restore(sess, restore_from=restore_from)

        # what is the value at level 2?
        val2_start_goal = network.predict_values([start], [goal], 2, sess)[0]
        print 'predicted level 2 value of start goal {}'.format(val2_start_goal)
        # assert that this is the same as splitting by the policy
        val1_start_mid = network.predict_values([start], [mid], 1, sess)
        val1_mid_goal = network.predict_values([mid], [goal], 1, sess)
        val2_with_mid = val1_start_mid + val1_mid_goal
        print 'mid point at level 2 is {}'.format(mid)
        print 'predicted level 2 value of start goal going through mid {}'.format(val2_with_mid)

        # what is the minimal value with for loop of 2 level 1s?
        val1_start_mid = network.predict_values([start]*len(points), points, 1, sess)
        val1_mid_goal = network.predict_values(points, [goal] * len(points), 1, sess)
        val1 = val1_start_mid + val1_mid_goal
        better_options = [i for i, f in enumerate([float(v) for v in val1]) if f < float(val2_start_goal)]
        print 'better points discovered ({}):'.format(len(better_options))
        for i in better_options:
            print('x: {} y: {} \t\t val: {}'.format(points[i][0], points[i][1], val1[i]))

        argmin_i = int(np.argmin(val1))
        print 'minimal value {} at x: {} y: {}'.format(val1[argmin_i], points[argmin_i][0], points[argmin_i][1])

        with open('/data/analysis/a.txt', 'w') as f:
            lines = ['{} {}{}'.format(points[i][0], points[i][1], os.linesep) for i in better_options]
            f.writelines(lines)




