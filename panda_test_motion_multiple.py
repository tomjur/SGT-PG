import numpy as np
import time
import random
import matplotlib.pyplot as plt

from panda_scene_manager import PandaSceneManager


panda_scene_manager = PandaSceneManager(use_ui=False)
lower = np.array(panda_scene_manager.joints_lower_bounds)
upper = np.array(panda_scene_manager.joints_upper_bounds)
sleep_time = 0.0
max_steps = 1000


max_forces = (500., 500., 500., 500., 50., 50., 50., 1., 1.)
# max_forces = (500., 500., 500., 500., 500., 500., 500., 10., 10.)
position_coeff = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3)
velocity_coeff = (1., 1., 1., 1., 1., 1., 1., 1., 1.)
# max_jerk = (100., 100., 100., 100., 100., 100., 100., 100., 100.)


def _print_robot(target):
    joint_positions, joint_velocities = panda_scene_manager.get_robot_state()
    is_collision = panda_scene_manager.is_collision()
    distance = np.linalg.norm(np.array(target) - np.array(joint_positions))
    velocity = np.linalg.norm(joint_velocities)
    print('target {}'.format(target))
    print('robot positions {}'.format(joint_positions))
    print('robot velocities {}'.format(joint_velocities))
    print('is collision {}'.format(is_collision))
    print('distance from target: {}'.format(distance))
    print('total velocity {}'.format(velocity))


def get_random_joints(upper, lower):
    random_relative = [random.random() for _ in range(9)]
    return [r * lower[i] + (1.-r) * upper[i] for i, r in enumerate(random_relative)]



for i in range(1000):
    joints1 = get_random_joints(upper, lower)
    joints2 = get_random_joints(upper, lower)

    # print('required initial joints: {}'.format(joints1))
    # print('required terminal joints: {}'.format(joints2))
    # print('initial distance: {}'.format(np.linalg.norm(np.array(joints1) - np.array(joints2))))

    panda_scene_manager.change_robot_joints(joints1)
    is_collision = panda_scene_manager.simulation_step()[1]
    if is_collision:
        # print('started in collision')
        # _print_robot(joints2)
        continue
    time.sleep(sleep_time)

    _, free_length, collision_length, visited_states = panda_scene_manager.walk_small_segment(
    joints2, time_between_frames=sleep_time, max_steps=max_steps
        # joints2, time_between_frames=sleep_time,  max_forces=max_forces, position_coeff=position_coeff,
        # velocity_coeff=velocity_coeff
    )
    if len(visited_states) >= max_steps:
        print('{}: max steps reached {} to {}'.format(i, joints1, joints2))

    # print('free length {} collision length {}'.format(free_length, collision_length))
    print('{}: is colision {} number of steps {}'.format(i, collision_length > 0., len(visited_states)))
    print('')

