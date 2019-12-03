

class EpisodeRunnerSequential:
    def __init__(self, config, game, policy_function):
        self.config = config
        self.game = game
        self.policy_function = policy_function

        self.fixed_start_goal_pairs = self.game.get_fixed_start_goal_pairs()

    def play_fixed_episodes(self, is_train=False):
        return self.play_episodes(self.fixed_start_goal_pairs, is_train)

    def play_random_episodes(self, number_of_episodes, is_train):
        start_goal_pairs = self.game.get_free_start_goals(number_of_episodes)
        return self.play_episodes(start_goal_pairs, is_train)

    def play_episodes(self, start_goal_pairs, is_train):
        return self.game.run_episodes(start_goal_pairs, is_train, self.policy_function)
