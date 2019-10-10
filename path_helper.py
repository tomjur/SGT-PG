import os


def get_base_directory():
    return os.path.join(os.getcwd(), 'data')


def get_config_directory():
    return os.path.join(get_base_directory(), 'config')


def init_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
