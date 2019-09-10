import os
import yaml

from docker_path_helper import get_base_directory, get_config_directory


def overload_config_by_scenario(config):
    scenario = config['general']['scenario']
    is_point_robot = 'point_robot' in scenario
    config['general']['is_point_robot'] = is_point_robot
    config['model']['state_size'] = 2 if is_point_robot else 4

    config['general']['state_bounds'] = ((-1., -1.), (1., 1.)) if is_point_robot else (
        (0., 0., 0., 0.), (1., 1., 1., 1.)
    )

    params_file = os.path.join(get_base_directory(), 'scenario_params', scenario)
    if scenario != 'vision':
        params_file = os.path.join(params_file, 'params.pkl')
    config['general']['params_file'] = params_file
    config['model']['consider_image'] = scenario == 'vision'


def read_config(config_path=None, overload_config=True):
    if config_path is None:
        config_path = os.path.join(get_config_directory(), 'config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        if overload_config:
            overload_config_by_scenario(config)
        print('------------ Config ------------')
        print(yaml.dump(config))
        return config


def copy_config(config, copy_to):
    with open(copy_to, 'w') as f:
        yaml.dump(config, f)
