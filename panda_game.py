import numpy as np

from panda_scene_manager import PandaSceneManager
from rl_interface import AbstractMotionPlanningGame


class PandaGame(AbstractMotionPlanningGame):
    def __init__(self, config):
        AbstractMotionPlanningGame.__init__(self, config)
        self.panda_scene_manager = PandaSceneManager(use_ui=False)
        if 'no_obs' in self.params_file:
            obstacles_definitions_list = []
        else:
            with open(self.params_file, 'r') as params_file:
                obstacles_definitions_list = params_file.readlines()
        self.panda_scene_manager.add_obstacles(obstacles_definitions_list)

    def can_recover_from_failed_movement(self):
        return False

    def check_terminal_segment(self, segment):
        start, end = segment
        is_start_valid = self.is_free_state(start)
        new_state = start
        collision_length = 0.0
        is_collision = False
        while not self.panda_scene_manager.is_close(new_state, end):
            new_state, is_collision = self.panda_scene_manager.single_step_move_all_joints_by_position(end)
            if is_collision:
                collision_length = np.linalg.norm(new_state - end)
                break
        if is_collision:
            is_goal_valid = False
        else:
            is_goal_valid = self.is_free_state(end)
        return is_start_valid, is_goal_valid, collision_length

    def is_free_state(self, state):
        self.panda_scene_manager.change_robot_joints(state)
        self.panda_scene_manager.simulation_step()
        traj = self.panda_scene_manager.reach_joint_positions(state, 100)
        (start_joints, start_velocities), is_collision = traj[-1]
        if not self.panda_scene_manager.is_close(start_joints, state):
            return False
        if is_collision:
            return False
        return True

    def _get_state_bounds(self):
        return self.panda_scene_manager.joints_lower_bounds, self.panda_scene_manager.joints_upper_bounds
