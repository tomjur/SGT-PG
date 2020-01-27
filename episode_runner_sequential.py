

class EpisodeRunnerSequential:
    def __init__(self, config, game, curriculum_coefficient=None):
        self.config = config
        self.game = game
        self.curriculum_coefficient = curriculum_coefficient
        self.repeat_train_trajectories = self.config['model']['repeat_train_trajectories']

        self.fixed_start_goal_pairs = self.game.get_fixed_start_goal_pairs()

    def play_fixed_episodes(self, is_train=False):
        return self.play_episodes(self.fixed_start_goal_pairs, is_train)

    def play_random_episodes(self, number_of_episodes, is_train):
        number_of_start_goal_pairs = number_of_episodes
        if is_train and self.repeat_train_trajectories is not None:
            number_of_start_goal_pairs = -(-number_of_episodes // self.repeat_train_trajectories)  # does int division on with
                                                                                           # ceiling instead of floor
        start_goal_pairs = self.game.get_free_start_goals(number_of_start_goal_pairs, self.curriculum_coefficient)
        if is_train and self.repeat_train_trajectories is not None:
            start_goal_pairs_ = []
            for _ in range(self.repeat_train_trajectories):
                for s, g in start_goal_pairs:
                    new_pair = (s.copy(), g.copy())
                    start_goal_pairs_.append(new_pair)
            start_goal_pairs = start_goal_pairs_[:number_of_episodes]
        return self.play_episodes(start_goal_pairs, is_train)

    def play_episodes(self, start_goal_pairs, is_train):
        return self.game.run_episodes(start_goal_pairs, is_train)
