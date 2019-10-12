import matplotlib.pyplot as plt
import os
import pickle
import random

from path_helper import get_base_directory, init_dir
from point_robot_manager import PointRobotManager

scenario = 'point_robot_box_0.2'
# scenario = 'point_robot_box_0.8'
# scenario = 'point_robot_easy2_transposed'
# scenario = 'point_robot_easy2'
# scenario = 'point_robot_hard_corridors'

model_name = '2019_10_10_14_44_02_to_remove'
cycles = ['-1']
source_trajectories = 'test_trajectories'
num_trajectories = -1
# num_trajectories = 1000


def parse_trajectory_line(line):
    line = line.replace('[', '').replace(']', '').replace(os.linesep, '')
    parts = [p for p in line.split(' ') if len(p) > 0]
    assert len(parts) == 2

    x1 = float(parts[0])
    y1 = float(parts[1])
    return x1, y1


trajectories_dir = os.path.join(get_base_directory(), scenario, source_trajectories, model_name)
output_dir = os.path.join(get_base_directory(), scenario, 'visualizations', source_trajectories, model_name)

obstacles_definitions_location = os.path.join(get_base_directory(), 'scenario_params', scenario, 'params.pkl')
with open(obstacles_definitions_location, 'rb') as pickle_file:
    obstacles_definitions = pickle.load(pickle_file)

for cycle in cycles:
    cycle_output_dir = os.path.join(output_dir, cycle)
    init_dir(cycle_output_dir)
    point_robot_manager = PointRobotManager(obstacles_definitions)

    trajectory_file = os.path.join(trajectories_dir, '{}.txt'.format(cycle))
    with open(trajectory_file) as f:
        all_lines = f.readlines()
    all_lines = ''.join(all_lines)
    data_by_path = all_lines.split('path_id_')
    data_by_path = [d for d in data_by_path if len(d) > 0]
    if 0 < num_trajectories < len(data_by_path):
        # sample if the number of trajectories is bigger than zero and less than total number of trajectories
        data_by_path = random.sample(data_by_path, num_trajectories)

    for d in data_by_path:
        lines = d.split(os.linesep)
        path_id = lines[0]
        status = lines[1]
        trajectory = [parse_trajectory_line(l) for l in lines[2:] if len(l) > 0]
        output_file_location = os.path.join(cycle_output_dir, '{}_{}.png'.format(path_id, status))
        point_robot_manager.plot(paths=[trajectory]).savefig(output_file_location)
        plt.clf()
        print('printed path_id {} ({}), trajectory length {}'.format(path_id, status, len(trajectory)))
