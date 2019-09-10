import os
import numpy as np
import cPickle as pickle

from docker_path_helper import init_dir


class ValueFunctionPrinter:

    def __init__(self, config, network, model_name):
        self.config = config
        self.network = network
        self.model_name = model_name

        value_functions_dir = os.path.join('data', self.config['general']['scenario'], 'value_functions', model_name)
        init_dir(value_functions_dir)
        self.value_functions_dir = value_functions_dir
        self.per_axis_states = 201

    def print_single_level(self, sess, goal_state, level):
        start_inputs = self._get_grid_states()
        goal_inputs = np.array([goal_state] * len(start_inputs))
        q_values = self.network.predict(start_inputs, goal_inputs, level, sess)
        q_values = q_values.reshape(self.per_axis_states, self.per_axis_states)
        file_location = os.path.join(self.value_functions_dir, 'goal_state_{}_level_{}.txt'.format(goal_state, level))
        with open(file_location, 'w') as result_file:
            pickle.dump(q_values.tolist(), result_file)

    def draw_all_levels(self, sess, goal_state):
        import matplotlib.pyplot as plt
        start_inputs = self._get_grid_states()
        goal_inputs = np.array(goal_state) * len(start_inputs)
        for level in range(self.config['model']['levels']):
            q_values = self.network.predict(start_inputs, goal_inputs, level, sess)
            q_values = q_values.reshape(self.per_axis_states, self.per_axis_states)
            file_location = os.path.join(self.value_functions_dir, 'level_{}.png'.format(level))
            fig = plt.figure(1, dpi=90)
            ax = fig.add_subplot(111)
            ax.imshow(q_values, cmap='viridis')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_aspect(1)
            fig.colorbar()
            fig.savefig(file_location)

    def _get_grid_states(self):
        step_size = 2. / (self.per_axis_states-1)
        # by discretization
        return [
            np.array([-1.0 + i * step_size, -1.0 + j * step_size])
            for i in range(self.per_axis_states)
            for j in range(self.per_axis_states)
        ]
