import numpy as np
import time
import random
import matplotlib.pyplot as plt

from panda_scene_manager import PandaSceneManager


panda_scene_manager = PandaSceneManager(use_ui=True)
lower = np.array(panda_scene_manager.joints_lower_bounds)
upper = np.array(panda_scene_manager.joints_upper_bounds)
sleep_time = 0.05


max_forces = (500., 500., 500., 500., 500., 500., 500., 10., 10.)
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


# joints1 = 0.3 * lower + 0.7 * upper
# joints2 = 0.5 * lower + 0.5 * upper
# alpha = 0.95
# joint_index = 3
# joints2[joint_index] = alpha * lower[joint_index] + (1.-alpha) * upper[joint_index]

joints1 = get_random_joints(upper, lower)
joints2 = get_random_joints(upper, lower)

print('required initial joints: {}'.format(joints1))
print('required terminal joints: {}'.format(joints2))
print('initial distance: {}'.format(np.linalg.norm(np.array(joints1) - np.array(joints2))))


panda_scene_manager.change_robot_joints(joints1)
is_collision = panda_scene_manager.simulation_step()[1]
if is_collision:
    print('started in collision')
    _print_robot(joints2)
    exit()
time.sleep(sleep_time)

# visited_states = panda_scene_manager.bounded_jerk_motion_model_predictive_control(
#     joints2, max_jerk, max_forces, position_coeff, velocity_coeff, sleep_time
# )[-1]

_, free_length, collision_length, visited_states = panda_scene_manager.walk_small_segment(
    joints2, time_between_frames=sleep_time,  max_forces=max_forces, position_coeff=position_coeff,
    velocity_coeff=velocity_coeff
)

print('free length {} collision length {}'.format(free_length, collision_length))
print('number of steps {}'.format(len(visited_states)))
print('')

for j, v in visited_states:
    print('joints {}'.format(j))
    print('velocity {}'.format(v))
    print('')

print('required terminal joints: {}'.format(joints2))



for i in range(9):
    data = [j[i] for j, v in visited_states]
    plt.plot(data)
    plt.show()

