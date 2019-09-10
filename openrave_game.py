import numpy as np
import os

from openrave_manager import OpenraveManager
from rl_interface import AbstractMotionPlanningGame


class OpenraveGame(AbstractMotionPlanningGame):
    def __init__(self, config):
        AbstractMotionPlanningGame.__init__(self, config)
        self.openrave_manager = OpenraveManager(config['openrave']['segment_validity_step'])
        params_file = config['general']['params_file']
        if not os.path.isdir(params_file):
            if params_file is not None:
                # we have a single params file - just load it
                self.openrave_manager.set_params(params_file)

    def check_terminal_segment(self, segment):
        return self.openrave_manager.check_segment_validity(segment[0], segment[1], is_virtual=True)

    def is_free_state(self, state):
        return self.openrave_manager.is_valid(state, is_virtual=True)
