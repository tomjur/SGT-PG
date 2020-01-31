import os
import numpy as np

from panda_game_subgoal import PandaGameSubgoal
from panda_scene_manager import PandaSceneManager
from path_helper import get_base_directory, deserialize_uncompress

scenario = 'panda_easy'

angle = -155
path_ids = [86]
test_trajectories_file = '/home/tom/sgt-pg-results/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level2_all.txt'


def show_trajectory(endpoints, panda_scene_manager):
    # set the robot to the initial location
    panda_scene_manager.change_robot_joints(endpoints[0])
    panda_scene_manager.set_camera(3.5, angle, -20, [0., 0., 0.])
    panda_scene_manager.simulation_step()
    # assert the teleportation was successful
    assert panda_scene_manager.is_close(endpoints[0])
    assert not panda_scene_manager.is_collision()
    sum_free, sum_collision = panda_scene_manager.smooth_walk(endpoints[-1], max_target_distance=1., sensitivity=0.01)
    print('sum collision {}'.format(sum_collision))
    assert sum_collision > 0.0



test_trajectories_by_path_id = deserialize_uncompress(test_trajectories_file)

panda_scene_manager = PandaSceneManager.get_scene_manager(scenario=scenario, use_ui=True)


for path_id in path_ids:
    is_successful = test_trajectories_by_path_id[path_id][1]
    print('path_id {} is successful? {}'.format(path_id, is_successful))
    if is_successful:
        print('showing trajectory')
        endpoints = test_trajectories_by_path_id[path_id][0]
        non_virtual_endpoints = [PandaGameSubgoal.virtual_to_real_state(e, panda_scene_manager) for e in endpoints]

        show_trajectory(non_virtual_endpoints, panda_scene_manager)
    else:
        print('cannot display trajectory')
