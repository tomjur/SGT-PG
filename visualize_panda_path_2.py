import os
import numpy as np

from panda_game_subgoal import PandaGameSubgoal
from panda_scene_manager import PandaSceneManager
from path_helper import get_base_directory, deserialize_uncompress

scenario = 'panda_poles'


level_1_test_trajectories_file = '/home/tom/sgt-pg-results/sgt/poles/sgt-poles-2/data/sgt/panda_poles/test_trajectories/2020_02_05_08_37_13/level1_all.txt'
level_2_test_trajectories_file = '/home/tom/sgt-pg-results/sgt/poles/sgt-poles-2/data/sgt/panda_poles/test_trajectories/2020_02_05_08_37_13/level2_all.txt'
level_3_test_trajectories_file = '/home/tom/sgt-pg-results/sgt/poles/sgt-poles-2/data/sgt/panda_poles/test_trajectories/2020_02_05_08_37_13/level3_all.txt'
# level_1_test_trajectories_file = '/home/tom/sgt-pg-results/sgt/poles/sgt-poles-1/data/sgt/panda_poles/test_trajectories/2020_02_05_08_37_16/level1_all.txt'
# level_2_test_trajectories_file = '/home/tom/sgt-pg-results/sgt/poles/sgt-poles-1/data/sgt/panda_poles/test_trajectories/2020_02_05_08_37_16/level2_all.txt'
# level_3_test_trajectories_file = '/home/tom/sgt-pg-results/sgt/poles/sgt-poles-1/data/sgt/panda_poles/test_trajectories/2020_02_05_08_37_16/level3_all.txt'

# scenario = 'panda_easy'
#
#
# level_1_test_trajectories_file = '/home/tom/sgt-pg-results/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level1_all.txt'
# level_2_test_trajectories_file = '/home/tom/sgt-pg-results/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level2_all.txt'
# level_3_test_trajectories_file = '/home/tom/sgt-pg-results/sgt-e1/data/sgt/panda_easy/test_trajectories/2020_01_29_07_58_09/level3_all.txt'



panda_scene_manager = PandaSceneManager.get_scene_manager(scenario=scenario, use_ui=False)


def parse_data(test_trajectories_file, panda_scene_manager):
    results = {}
    test_trajectories_by_path_id = deserialize_uncompress(test_trajectories_file)
    for path_id in test_trajectories_by_path_id:
        is_successful = test_trajectories_by_path_id[path_id][1]
        endpoints = test_trajectories_by_path_id[path_id][0]
        non_virtual_endpoints = [PandaGameSubgoal.virtual_to_real_state(e, panda_scene_manager) for e in endpoints]
        start = non_virtual_endpoints[0]
        goal = non_virtual_endpoints[-1]
        results[path_id] = (start, goal, is_successful)
    return results

data1 = parse_data(level_1_test_trajectories_file, panda_scene_manager)
data2 = parse_data(level_2_test_trajectories_file, panda_scene_manager)
data3 = parse_data(level_3_test_trajectories_file, panda_scene_manager)
assert len(data1.keys()) == len(data2.keys()) == len(data3.keys())
for path_id in data1:
    assert np.all(data1[path_id][0] == data2[path_id][0])
    assert np.all(data1[path_id][0] == data3[path_id][0])
    assert np.all(data1[path_id][1] == data2[path_id][1])
    assert np.all(data1[path_id][1] == data3[path_id][1])


def compare_results(name1, name2, data1, data2):
    result = []
    for path_id in data1:
        if data1[path_id][2] and not data2[path_id][2]:
            print("{} managed to plan for {} and {} didn't".format(name1, path_id, name2))
            result.append(path_id)
    return result

_1_over_2 = compare_results('level 1', 'level 2', data1, data2)
_2_over_1 = compare_results('level 2', 'level 1', data2, data1)
_1_over_3 = compare_results('level 1', 'level 3', data1, data3)
_3_over_1 = compare_results('level 3', 'level 1', data3, data1)
_2_over_3 = compare_results('level 2', 'level 3', data2, data3)
_3_over_2 = compare_results('level 3', 'level 2', data3, data2)


for path_id in _3_over_1:
    current_distance = np.linalg.norm(data2[path_id][0] - data2[path_id][1])
    print('path id {} distance {}'.format(path_id, current_distance))
