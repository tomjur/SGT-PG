import os
import time

import numpy as np

from panda_game_subgoal import PandaGameSubgoal
from panda_scene_manager import PandaSceneManager
from path_helper import get_base_directory, deserialize_uncompress

# scenario = 'panda_poles'
#
# angle = -75
# over_angle = -40
# path_ids = [32]
# test_trajectories_file = '/home/tom/sgt-pg-results/sgt/poles/sgt-poles-2/data/sgt/panda_poles/test_trajectories/2020_02_05_08_37_13/level3_all.txt'

scenario = 'panda_easy'
over_angle = -20

angle = -75
# angle = 200
path_ids = [0]
test_trajectories_file = '/home/tom/sgt-pg-results/sgt/easy/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level2_challenging.txt'

# angle = 15
# path_ids = [33]
# test_trajectories_file = '/home/tom/sgt-pg-results/sgt/easy/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level2_all.txt'

# angle = -155
# path_ids = [79]
# test_trajectories_file = '/home/tom/sgt-pg-results/sgt/easy/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level2_all.txt'

# angle = -155
# path_ids = [93]
# test_trajectories_file = '/home/tom/sgt-pg-results/sgt/easy/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level2_all.txt'

# angle = -155
# path_ids = [31]
# test_trajectories_file = '/home/tom/sgt-pg-results/sgt/easy/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level2_all.txt'

# angle = -155
# path_ids = [86]
# test_trajectories_file = '/home/tom/sgt-pg-results/sgt/easy/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level2_all.txt'


def show_trajectory(endpoints, panda_scene_manager):
    # set the robot to the initial location
    panda_scene_manager.change_robot_joints(endpoints[0])
    panda_scene_manager.set_camera(3.5, angle, over_angle, [0., 0., 0.])
    panda_scene_manager.simulation_step()
    # assert the teleportation was successful
    assert panda_scene_manager.is_close(endpoints[0])
    assert not panda_scene_manager.is_collision()
    # while True:
    #     time.sleep(1.)
    # walk the segment
    distance_traveled = 0.0
    for i in range(len(endpoints)-1):
        sum_free, sum_collision = panda_scene_manager.smooth_walk(endpoints[i+1], max_target_distance=1., sensitivity=0.01)
        assert sum_collision == 0.0
        distance_traveled += sum_free
        # while i == 8:
        #     time.sleep(1.)
    print('total distance {}'.format(distance_traveled))



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
