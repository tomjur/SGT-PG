class AbstractMotionPlanningGameSubgoal:
    def get_fixed_start_goal_pairs(self, challenging=False):
        assert False

    def get_start_goals(self, number_of_pairs, curriculum_coefficient, get_free_states):
        assert False

    def test_predictions(self, predictions):
        assert False

    def is_free_state(self, state):
        assert False

    def get_state_size(self):
        assert False
