from path_helper import deserialize_uncompress

file_path = '/home/tom/SGT-PG/data/sgt/disks/test_trajectories/2020_04_19_14_42_28/level1_all.txt'

paths = deserialize_uncompress(file_path)
for path_id in paths:
    print(f'path id {path_id}, is successful {paths[path_id][1]}')
    print(f'path')
    print(paths[path_id][0])
