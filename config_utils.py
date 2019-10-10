import os
import yaml

from path_helper import get_config_directory


def read_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(get_config_directory(), 'config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        print('------------ Config ------------')
        print(yaml.dump(config))
        return config


def copy_config(config, copy_to):
    with open(copy_to, 'w') as f:
        yaml.dump(config, f)
