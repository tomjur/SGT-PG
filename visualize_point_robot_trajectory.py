import os
import cPickle as pickle
import glob

from docker_path_helper import get_base_directory, init_dir
from point_robot_manager import PointRobotManager

if __name__ == '__main__':
    scenario = 'point_robot_hard_corridors'
    # scenario = 'point_robot_easy'
    model_name = '2019_05_15_12_03_55'
    global_step = -1

    def get_from_pickle(filepath):
        with open(filepath, 'r') as f:
            return pickle.load(f)

    global_step = str(global_step)
    paths_dir = os.path.join(get_base_directory(), scenario, 'trajectories', model_name, global_step)
    path_files = [f for f in glob.glob(paths_dir + '/*.p') if '_motion_planner_' not in f]

    image_dir = os.path.join('data', 'temp', model_name, global_step)
    init_dir(image_dir)

    params_file = os.path.join(get_base_directory(), 'scenario_params', scenario, 'params.pkl')
    point_robot_manager = PointRobotManager(get_from_pickle(params_file))

    for path_file in path_files:
        input_filename = [part for part in path_file.split('/') if len(part) > 0][-1]
        reference_path_file = os.path.join(paths_dir, input_filename.replace('_', '_motion_planner_'))

        planned_path = get_from_pickle(path_file)
        planned_path = [segment[0][0].tolist() for segment in planned_path] + [planned_path[-1][0][1].tolist()]
        # planned_path = [p[0] for p in planned_path] + [planned_path[-1][1]]
        motion_planner_path = get_from_pickle(reference_path_file)

        f = point_robot_manager.plot([motion_planner_path, planned_path])

        output_filename = input_filename.replace('.p', '.png')
        image_path = os.path.join(image_dir, output_filename)
        f.savefig(image_path)
        f.clear()
