import numpy as np


class AbstractMotionPlanningGameSubgoal:
    def __init__(self, config):
        self.config = config
        lower, upper = self._get_state_bounds()
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.state_size = len(lower)

    def get_valid_states(self, states):
        return np.maximum(self.lower, np.minimum(self.upper, states))

    def get_random_state(self):
        return np.random.uniform(self.lower, self.upper, self.state_size)

    def get_free_random_state(self):
        while True:
            state = self.get_random_state()
            if self.is_free_state(state):
                return state

    def test_predictions(self, predictions):
        assert False

    def is_free_state(self, state):
        assert False

    def _get_state_bounds(self):
        return (0, ), (0, )

    def get_fixed_start_goal_pairs(self):
        assert False

    def get_free_states(self, number_of_states):
        assert False
