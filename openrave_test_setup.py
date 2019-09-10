import tensorflow as tf
import yaml

from docker_path_helper import *
from openrave_manager import OpenraveManager
from potential_point import PotentialPoint


if __name__ == '__main__':
    print(os.system("nvidia-smi"))

    is_gpu = tf.test.is_gpu_available()
    print 'has gpu result {}'.format(is_gpu)

    config_path = os.path.join(get_config_directory(), 'config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file, Loader=yaml.SafeLoader)
    potential_points = PotentialPoint.from_config(config)
    openrave_manager = OpenraveManager(0.01)
    random_joints = openrave_manager.get_random_joints()

    print 'random joints result {}'.format(random_joints)
