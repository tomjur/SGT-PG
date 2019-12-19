from panda_game_sequential import PandaGameSequential
from panda_game_subgoal import PandaGameSubgoal

scenarios = ['panda_no_obs', 'panda_easy', 'panda_hard']

expected = 100

for scenario in scenarios:
    game = PandaGameSubgoal(scenario, max_cores=1)
    fixed_pairs = game.get_fixed_start_goal_pairs()
    assert len(fixed_pairs) == expected

    game = PandaGameSequential(scenario, None, None, None, max_cores=1)
    fixed_pairs = game.get_fixed_start_goal_pairs()
    assert len(fixed_pairs) == expected

print('done')