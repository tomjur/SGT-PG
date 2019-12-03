class AbstractMotionPlanningGameSequential:
    def __init__(self, config):
        self.config = config
        self.state_size, self.action_size = self.get_sizes()

    def get_fixed_start_goal_pairs(self):
        assert False

    def get_free_start_goals(self, number_of_episodes):
        assert False

    def run_episodes(self, start_goal_pairs, is_train, policy_function):
        assert False

    def get_sizes(self):
        return None, None
