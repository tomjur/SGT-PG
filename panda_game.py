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

    def check_terminal_segment(self, segment):
        start, end = segment
        # check the end points
        truncated_start, truncated_distance_start = self._truncate_virtual_state(start)
        truncated_end, truncated_distance_end = self._truncate_virtual_state(end)

        start_ = self._virtual_to_real_state(truncated_start)
        end_ = self._virtual_to_real_state(truncated_end)

        # partition to waypoints
        waypoints = self._get_waypoints(start_, end_)
        # we want to get is_goal_valid by moving in the "segment" [end_, end_]
        waypoints.append(end_)

        # check each waypoint
        sum_free = 0.0
        sum_collision = 0.0
        is_start_valid = truncated_distance_start == 0.0
        is_goal_valid = truncated_distance_end == 0.0
        for waypoint_index in range(len(waypoints)-1):
            is_start_waypoint_valid, free_length, collision_length = self._walk_small_segment(
                waypoints[waypoint_index], waypoints[waypoint_index+1]
            )
            if waypoint_index == 0:
                is_start_valid = is_start_valid and is_start_waypoint_valid
            if waypoint_index == len(waypoints)-2:
                is_goal_valid = is_goal_valid and is_start_waypoint_valid
            sum_free += free_length
            sum_collision += collision_length

        is_start_free = self.is_free_state(truncated_start)
        is_end_free = self.is_free_state(truncated_end)
        sum_collision += truncated_distance_start + truncated_distance_end
        return is_start_free, is_end_free, sum_free, sum_collision

    def _get_waypoints(self, start, end):
        max_step = self.panda_scene_manager.position_sensitivity * 5000
        initial_distance = np.linalg.norm(end - start)
        num_steps = int(np.ceil(initial_distance / max_step))

        direction = end - start
        direction = direction / np.linalg.norm(direction)
        waypoints = [start + step*direction for step in np.linspace(0.0, initial_distance, num_steps + 1)]
        assert self.panda_scene_manager.is_close(start, waypoints[0])
        assert self.panda_scene_manager.is_close(end, waypoints[-1])
        return waypoints

    def _walk_small_segment(self, start, end):
        segment_length = np.linalg.norm(start - end)
        if not self.panda_scene_manager.is_close(start):
            self.panda_scene_manager.change_robot_joints(start)
            self.panda_scene_manager.simulation_step()
            self.panda_scene_manager.reach_joint_positions(start, max_steps=100, stop_on_collision=True)
        is_start_valid = self.panda_scene_manager.is_close(start) and not self.panda_scene_manager.is_collision()
        if not is_start_valid:
            # if the segment is not free
            return is_start_valid, 0.0, segment_length
        # this is the maximal allowed divergence
        allowed_distance = segment_length * 1.5
        is_free = True
        while not self.panda_scene_manager.is_close(end):
            (current_position, _), is_collision = self.panda_scene_manager.single_step_move_all_joints_by_position(end)
            current_position_ = np.array(current_position)
            if is_collision or np.linalg.norm(current_position_ - end) > allowed_distance:
                is_free = False
                break
        if is_free:
            free_length = segment_length
            collision_length = 0.0
        else:
            free_length = 0.0
            collision_length = segment_length
        return is_start_valid, free_length, collision_length

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
