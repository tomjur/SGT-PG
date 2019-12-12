import os
import pickle
import zlib


def get_base_directory():
    return os.path.join(os.getcwd(), 'data')


def get_config_directory():
    return os.path.join(get_base_directory(), 'config')


def init_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_scenario_dir(scenario):
    return os.path.join(get_base_directory(), 'scenario_params', scenario)


def get_params_from_scenario(scenario):
    return os.path.join(get_scenario_dir(scenario), 'params.pkl')


def get_start_goal_from_scenario(scenario):
    return os.path.join(get_scenario_dir(scenario), 'virtual-start-goal.txt')


def serialize_compress(data, path):
    with open(path, 'wb') as zf:
        compressed = zlib.compress(pickle.dumps(data))
        zf.write(compressed)


def deserialize_uncompress(path):
    with open(path, 'rb') as zf:
        compressed = zf.read()
        return pickle.loads(zlib.decompress(compressed))
