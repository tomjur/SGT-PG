class AbstractMotionPlanningGameSequential:
    def get_fixed_start_goal_pairs(self):
        assert False

    def get_free_start_goals(self, number_of_pairs, curriculum_coefficient):
        assert False

    def run_episodes(self, start_goal_pairs, is_train):
        assert False

    def get_state_space_size(self):
        assert False

    def get_action_space_size(self):
        assert False

    def get_number_of_workers(self):
        assert False
