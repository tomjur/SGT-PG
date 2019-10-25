import os
import numpy as np

from panda_scene_manager import PandaSceneManager
from path_helper import get_base_directory, deserialize_uncompress
from panda_game import PandaGame

# global_step = 22
global_step = 73
scenario = 'panda_no_obs'
model_name = '2019_10_24_15_36_55'
time_between_frames = 0.01

config = {'general': {'scenario': scenario}}


def show_trajectory(endpoints, panda_scene_manager):
    # reset the visualization
    panda_scene_manager.reset_simulation()
    PandaGame.add_obstacles(config, panda_scene_manager)
    # set the robot to the initial location
    panda_scene_manager.change_robot_joints(endpoints[0])
    panda_scene_manager.simulation_step()
    # assert the teleportation was successful
    assert panda_scene_manager.is_close(endpoints[0])
    assert not panda_scene_manager.is_collision()
    # walk the segment
    distance_traveled = 0.0
    for i in range(len(endpoints)-1):
        _, _, sum_free, sum_collision = panda_scene_manager.walk_between_waypoints(
            endpoints[i], endpoints[i+1], teleport_between_waypoints=False, time_between_frames=time_between_frames)
        assert sum_collision == 0.0
        distance_traveled += sum_free
    print('total distance {}'.format(distance_traveled))



test_trajectories_dir = os.path.join(get_base_directory(), scenario, 'test_trajectories', model_name)
test_trajectories_file = os.path.join(test_trajectories_dir, '{}.txt'.format(global_step))
test_trajectories_by_path_id = deserialize_uncompress(test_trajectories_file)

panda_scene_manager = PandaSceneManager(use_ui=True)

# fake_endpoints = [np.array([0.0]*9), np.array([0.4]*9), np.array([0.8]*9)]
# fake_endpoints = [PandaGame.virtual_to_real_state(e, panda_scene_manager) for e in fake_endpoints]
# show_trajectory(fake_endpoints, panda_scene_manager)

for path_id in test_trajectories_by_path_id:
    is_successful = test_trajectories_by_path_id[path_id][1]
    print('path_id {} is successful? {}'.format(path_id, is_successful))
    if is_successful:
        print('showing trajectory')
        endpoints = test_trajectories_by_path_id[path_id][0]
        non_virtual_endpoints = [PandaGame.virtual_to_real_state(e, panda_scene_manager) for e in endpoints]

        show_trajectory(non_virtual_endpoints, panda_scene_manager)
    else:
        print('cannot display trajectory')
