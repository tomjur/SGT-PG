import os

from panda_game_subgoal import PandaGameSubgoal
from path_helper import get_start_goal_from_scenario

number_of_pairs = 100

# scenario = 'panda_no_obs'
# scenario = 'panda_easy'
# scenario = 'panda_hard'
scenario = 'panda_poles'

panda_game = PandaGameSubgoal(scenario, 8)
# existing_pairs = panda_game.get_fixed_start_goal_pairs()
# new_pairs = panda_game.get_free_start_goals(number_of_pairs - len(existing_pairs), None)
new_pairs = panda_game.get_free_start_goals(number_of_pairs, None)

print(len(new_pairs))

all_pairs = new_pairs
# all_pairs = existing_pairs + new_pairs
with open(get_start_goal_from_scenario(scenario), 'w') as f:
    for s, g in all_pairs:
        f.write('{}{}'.format(s.tolist(), os.linesep))
        f.write('{}{}'.format(g.tolist(), os.linesep))
print('done')
