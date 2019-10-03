import numpy as np

from panda_scene_manager import PandaSceneManager
from rl_interface import AbstractMotionPlanningGame


class PandaGame(AbstractMotionPlanningGame):
    def __init__(self, config):
        self.panda_scene_manager = PandaSceneManager(use_ui=False)
        AbstractMotionPlanningGame.__init__(self, config)
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
        # check the end points
        truncated_start, truncated_distance_start = self._truncate_virtual_state(start)
        truncated_end, truncated_distance_end = self._truncate_virtual_state(end)
        truncated_segment_distance = np.linalg.norm(truncated_start - truncated_end)

        is_start_free = self.is_free_state(truncated_start)
        is_end_free = self.is_free_state(truncated_end)

        # we already know it is free, but the side effect is that the robot moves to the start
        start_ = self._virtual_to_real_state(truncated_start)
        end_ = self._virtual_to_real_state(truncated_end)

        # before marking start as free, teleport the robot to the start again see it is free and close:
        if is_start_free and is_end_free:
            self.panda_scene_manager.change_robot_joints(start_)
            self.panda_scene_manager.simulation_step()
            traj = self.panda_scene_manager.reach_joint_positions(start_, 100)
            (start_joints, start_velocities), is_collision = traj[-1]
            if not self.panda_scene_manager.is_close(start_joints, start_) or is_collision:
                is_start_free = False

        if not (is_start_free and is_end_free):
            collision_length = truncated_segment_distance
            free_length = 0.0
        else:
            traj = self.panda_scene_manager.reach_joint_positions(end_, 100)
            last_free_waypoint_ = traj[-1][0][0]
            # last_free_waypoint_ = self.panda_scene_manager.slow_reach_joint_positions(end_)
            last_free_waypoint = self._real_to_virtual_state(last_free_waypoint_)
            if self.panda_scene_manager.is_close(last_free_waypoint_, end_):
                collision_length = 0.0
            else:
                collision_length = np.linalg.norm(last_free_waypoint - truncated_end)
            free_length = np.linalg.norm(last_free_waypoint - truncated_start)
        is_start_free = is_start_free and truncated_distance_start == 0.
        is_end_free = is_end_free and truncated_distance_end == 0.
        collision_length += truncated_distance_start + truncated_distance_end
        return is_start_free, is_end_free, free_length, collision_length

    @staticmethod
    def _truncate_virtual_state(state):
        truncated_state = np.maximum(np.minimum(state, 1.), -1.)
        truncated_distance = np.linalg.norm(state - truncated_state)
        return truncated_state, truncated_distance

    def is_free_state(self, state):
        if any(np.abs(state)) > 1.0:
            return False
        state_ = self._virtual_to_real_state(state)
        self.panda_scene_manager.change_robot_joints(state_)
        self.panda_scene_manager.simulation_step()
        traj = self.panda_scene_manager.reach_joint_positions(state_, 100)
        (start_joints, start_velocities), is_collision = traj[-1]
        if not self.panda_scene_manager.is_close(start_joints, state_):
            return False
        if is_collision:
            return False
        return True

    def _get_state_bounds(self):
        lower = np.array([-1.] * self.panda_scene_manager.number_of_joints)
        upper = np.array([1.] * self.panda_scene_manager.number_of_joints)
        return lower, upper

    def get_fixed_start_goal_pairs(self):
        # for now a long straight movement
        s = np.array(self.lower) * 0.8
        g = np.array(self.upper) * 0.8
        return [(s, g)]

    def _virtual_to_real_state(self, s):
        lower = np.array(self.panda_scene_manager.joints_lower_bounds)
        upper = np.array(self.panda_scene_manager.joints_upper_bounds)
        s_ = np.array(s)
        s_ = (upper - lower) * s_ + (upper + lower)
        s_ = s_ / 2.
        return s_

    def _real_to_virtual_state(self, s):
        lower = np.array(self.panda_scene_manager.joints_lower_bounds)
        upper = np.array(self.panda_scene_manager.joints_upper_bounds)
        denom = upper - lower
        s_ = np.array(s)
        s_ = 2 * s_ - (upper + lower)
        s_ = s_ / denom
        return s_
