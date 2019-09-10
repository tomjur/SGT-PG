import os
import pickle

import yaml

from config_utils import overload_config_by_scenario
from docker_path_helper import get_config_directory, get_base_directory
from openrave_manager import OpenraveManager


def get_path_filename(scenario, model_name, path_id, level):
    return os.path.join(
        get_base_directory(), scenario, 'planned_paths', model_name, 'path_id_{}_level_{}.txt'.format(path_id, level)
    )


def load_plan(scenario, model_name, path_id, level):
    path_filename = get_path_filename(scenario, model_name, path_id, level)
    with open(path_filename, 'r') as planned_path_file:
        return pickle.load(planned_path_file)


if __name__ == '__main__':
    # read the config
    config_path = os.path.join(get_config_directory(), 'config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        overload_config_by_scenario(config)

    # model_name = '2019_08_01_07_52_06'
    # model_name = '2019_08_01_06_23_43'
    model_name = '2019_08_01_10_12_54'
    path_id = '1'

    segment_verification = 0.001
    scenario = config['general']['scenario']

    plan = load_plan(scenario, model_name, path_id, 0)


    openrave_manager = OpenraveManager(segment_verification)
    openrave_manager.set_params(config['general']['params_file'])

    invalid_joints = [0., -0.7855, 0., 0., -1.3085]
    print 'config inside obstacle is valid?'
    print openrave_manager.is_valid(invalid_joints, is_virtual=False)

    for level in range(config['model']['levels']-1):
        plan = load_plan(scenario, model_name, path_id, level)
        print ''
        print 'level {}'.format(level)
        print 'is original path[0] -> path[-1] valid?'
        print openrave_manager.check_segment_validity(plan[0], plan[-1], is_virtual=True)

        all_valid = True
        for i in range(len(plan)-1):
            j0 = plan[i]
            j1 = plan[i+1]
            segment_validity = openrave_manager.check_segment_validity(j0, j1, is_virtual=True)
            print 'from {} to {} is valid? {}'.format(j0, j1, segment_validity)
            all_valid = all_valid and segment_validity

        print ''
        print 'path valid? {}'.format(all_valid)



