import glob
import os
import pickle
import yaml
import time
import numpy as np

from openrave_manager import OpenraveManager
from docker_path_helper import get_base_directory, get_config_directory
from openravepy import *

# global variables:

scenario = 'hard'
model_name = '2019_04_24_10_20_37'
global_step = '226035'
path_id = '0'
message = 'collision'
# message = 'success'

trajectories_dir = os.path.join(get_base_directory(), scenario, 'trajectories')
speed = 2000.0

trajectory_spheres_radius = 0.01
show_reference_spheres = True
show_reference_trajectory = True
repeat_in_loop = False


def create_sphere(id, radius, openrave_manager):
    body = RaveCreateKinBody(openrave_manager.env, '')
    body.SetName('sphere{}'.format(id))
    body.InitFromSpheres(np.array([[0.0]*3 + [radius]]), True)
    openrave_manager.env.Add(body, True)
    return body


def move_body(body, offset, theta):
    transformation_matrix = np.eye(4)

    translation = np.array(offset)
    rotation_matrix = np.array([
        [np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]
    ])
    transformation_matrix[:3, -1] = translation
    transformation_matrix[:3, :3] = rotation_matrix
    body.SetTransform(transformation_matrix)


def load_pickle(model_dir, header):
    search_key = os.path.join(model_dir, global_step, '{}_{}.p'.format(header, path_id))
    pickle_file = glob.glob(search_key)[0]
    return pickle.load(open(pickle_file))


def draw_sphere_at(joints, name, openrave_manager, color_np):
    pose = openrave_manager.get_target_pose(joints)
    pose = (pose[0], 0.0, pose[1])
    sphere = create_sphere(name, trajectory_spheres_radius, openrave_manager)
    move_body(sphere, pose, 0.0)
    sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color_np)


def draw_trajectory_spheres(trajectory, name, openrave_manager, colors_np):
    for i in range(len(trajectory)):
        draw_sphere_at(trajectory[i], '{}_{}'.format(name, i), openrave_manager, colors_np[i])


def get_detailed_trajectory(trajectory, openrave_manager):
    all = []
    for i in range(len(trajectory)-1):
        new_steps = openrave_manager.partition_segment(trajectory[i], trajectory[i+1])
        all.extend(new_steps)
    return all


def play(detailed_trajectory, openrave_manager):
    print 'len(trajectory) ', len(detailed_trajectory)
    for i in range(len(detailed_trajectory)):
        print 'i ', i
        openrave_manager.robot.SetDOFValues(detailed_trajectory[i])
        time.sleep(1/speed)


def main():
    config_path = os.path.join(get_config_directory(), 'config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        print('------------ Config ------------')
        print(yaml.dump(config))
    model_dir = os.path.join(trajectories_dir, model_name)
    openrave_manager = OpenraveManager(0.001)
    # get the parameters
    workspace_params_path = os.path.join(get_base_directory(), 'scenario_params', scenario, 'params.pkl')
    if workspace_params_path is not None:
        openrave_manager.set_params(workspace_params_path)
    openrave_manager.get_initialized_viewer()

    # load trajectories:
    planned = load_pickle(model_dir, message)
    original = load_pickle(model_dir, message + '_motion_planner')

    # from planned take the first joints of each segment, and the last segment's second joints to create a trajectory
    planned_traj = [list(s[0][0]) for s in planned] + [list(planned[-1][0][1])]
    planned_statuses = [s[1] for s in planned]
    assert len(planned_traj) == len(planned_statuses) + 1

    # add initial joints for both:
    planned_traj = [[0.0] + j for j in planned_traj]
    original_traj = [[0.0] + list(j) for j in original]

    # draw original spheres:
    if show_reference_spheres:
        original_sphere_colors = [np.array([1, 1, 1])] * len(original_traj)
        draw_trajectory_spheres(original_traj, 'original', openrave_manager, original_sphere_colors)

    # select colors for planned trajectory
    # [240, 100, 10] blue
    # [100, 204, 204] yellow
    planned_sphere_colors = [np.array([0, 0, 204])] * len(planned_traj)
    for i, status in enumerate(planned_statuses):
        if status != 1:
            planned_sphere_colors[i] = planned_sphere_colors[i+1] = np.array([50, 0, 0])
    draw_trajectory_spheres(planned_traj, 'planned', openrave_manager, planned_sphere_colors)

    # detailed trajectories:
    original_traj_detailed = []
    if show_reference_trajectory:
        original_traj_detailed = get_detailed_trajectory(original_traj, openrave_manager)
    planned_traj_detailed = get_detailed_trajectory(planned_traj, openrave_manager)

    while True:
        play(planned_traj_detailed, openrave_manager)
        play(original_traj_detailed, openrave_manager)
        if not repeat_in_loop:
            break

    # visualize the trajectory
    # trajectory = [[0.0] + list(j) for j in trajectory]

    # if display_start_goal_end_spheres:
    #     start = trajectory[0]
    #     end = trajectory[-1]
    #     pose_start = openrave_manager.get_target_pose(start)
    #     pose_start = (pose_start[0], 0.0, pose_start[1])
    #     pose_goal = (pose_goal[0], 0.0, pose_goal[1])
    #     pose_end = openrave_manager.get_target_pose(end)
    #     pose_end = (pose_end[0], 0.0, pose_end[1])
    #     start_sphere = create_sphere('start', trajectory_spheres_radius, openrave_manager)
    #     move_body(start_sphere, pose_start, 0.0)
    #     goal_sphere = create_sphere('goal', trajectory_spheres_radius, openrave_manager)
    #     move_body(goal_sphere, pose_goal, 0.0)
    #     end_sphere = create_sphere('end', trajectory_spheres_radius, openrave_manager)
    #
    #     start_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array([0, 0, 204]))
    #     goal_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array([240, 100, 10]))
    #     end_sphere.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(np.array([100, 204, 204]))
    #     move_body(end_sphere, pose_end, 0.0)
    #



# def get_coordinate(link_transforms, attached_link_index, initial_offset):
#     link_transform = link_transforms[attached_link_index]
#     coordinate = np.matmul(link_transform, np.append(initial_offset, [1.0]))
#     return coordinate[:3]

if __name__ == '__main__':
    main()
