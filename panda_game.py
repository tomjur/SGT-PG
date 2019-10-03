import numpy as np
import multiprocessing
import Queue
import time

from panda_scene_manager import PandaSceneManager
from rl_interface import AbstractMotionPlanningGame


class PandaGame(AbstractMotionPlanningGame):
    def __init__(self, config):
        self.panda_scene_manager = self.get_scene_manager(config)
        AbstractMotionPlanningGame.__init__(self, config)

        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()

        self.workers = [
            GameWorker(config, self.requests_queue, self.results_queue)
            for _ in range(self._get_number_of_workers(config))
        ]

        for w in self.workers:
            w.start()

    @staticmethod
    def _get_number_of_workers(config):
        # return 1
        # get configuration values
        test_episodes = config['general']['test_episodes']
        train_episodes_per_cycle = config['general']['train_episodes_per_cycle']
        repeat_train_trajectories = config['model']['repeat_train_trajectories']
        max_episodes = max(test_episodes, train_episodes_per_cycle * repeat_train_trajectories)
        # get available cpus
        cores = multiprocessing.cpu_count()
        max_cores_with_slack = max(cores - 2, 1)
        # the number of workers is the min between those
        return min(max_episodes, max_cores_with_slack)

    @staticmethod
    def get_scene_manager(config):
        panda_scene_manager = PandaSceneManager(use_ui=False)
        params_file = AbstractMotionPlanningGame.get_params_from_config(config)
        if 'no_obs' in params_file:
            obstacles_definitions_list = []
        else:
            with open(params_file, 'r') as f:
                obstacles_definitions_list = f.readlines()
        panda_scene_manager.add_obstacles(obstacles_definitions_list)
        return panda_scene_manager

    @staticmethod
    def _truncate_virtual_state(state):
        truncated_state = np.maximum(np.minimum(state, 1.), -1.)
        truncated_distance = np.linalg.norm(state - truncated_state)
        return truncated_state, truncated_distance

    def is_free_state(self, state):
        return self._is_free_state_in_manager(state, self.panda_scene_manager)

    @staticmethod
    def _is_free_state_in_manager(state, panda_scene_manager):
        if any(np.abs(state)) > 1.0:
            return False
        state_ = PandaGame._virtual_to_real_state(state, panda_scene_manager)
        panda_scene_manager.change_robot_joints(state_)
        panda_scene_manager.simulation_step()
        traj = panda_scene_manager.reach_joint_positions(state_, 100)
        (start_joints, start_velocities), is_collision = traj[-1]
        if not panda_scene_manager.is_close(start_joints, state_):
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

    def check_terminal_segments(self, cost_queries):
        # put all requests
        for path_id in cost_queries:
            path_requests = (path_id, tuple(cost_queries[path_id]))
            self.requests_queue.put(path_requests)

        # pull all responses
        results = {}
        for _ in range(len(cost_queries)):
            response = self.results_queue.get(block=True)
            path_id, path_results = response
            results[path_id] = {t[0]: t[1] for t in path_results}
        return results

    @staticmethod
    def _virtual_to_real_state(s, panda_scene_manager):
        lower = np.array(panda_scene_manager.joints_lower_bounds)
        upper = np.array(panda_scene_manager.joints_upper_bounds)
        s_ = np.array(s)
        s_ = (upper - lower) * s_ + (upper + lower)
        s_ = s_ / 2.
        return s_

    @staticmethod
    def _real_to_virtual_state(s, panda_scene_manager):
        lower = np.array(panda_scene_manager.joints_lower_bounds)
        upper = np.array(panda_scene_manager.joints_upper_bounds)
        denom = upper - lower
        s_ = np.array(s)
        s_ = 2 * s_ - (upper + lower)
        s_ = s_ / denom
        return s_


class GameWorker(multiprocessing.Process):
    def __init__(self, config, requests_queue, results_queue):
        multiprocessing.Process.__init__(self)
        self.config = config
        self.requests_queue = requests_queue
        self.results_queue = results_queue

        self.panda_scene_manager = None

    def run(self):
        self.panda_scene_manager = PandaGame.get_scene_manager(self.config)
        while True:
            try:
                request = self.requests_queue.get(block=True, timeout=0.001)
                path_id, cost_queries = request
                path_cost_results = [
                    (i, self.check_terminal_segment(start, end))
                    for i, start, end in cost_queries
                ]
                response = (path_id, path_cost_results)
                self.results_queue.put(response)
            except Queue.Empty:
                time.sleep(1.0)

    def check_terminal_segment(self, start, end):
        # check the end points
        truncated_start, truncated_distance_start = PandaGame._truncate_virtual_state(start)
        truncated_end, truncated_distance_end = PandaGame._truncate_virtual_state(end)

        start_ = PandaGame._virtual_to_real_state(truncated_start, self.panda_scene_manager)
        end_ = PandaGame._virtual_to_real_state(truncated_end, self.panda_scene_manager)

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

        sum_collision += truncated_distance_start + truncated_distance_end
        return start, end, is_start_valid, is_goal_valid, sum_free, sum_collision

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
            current_distance = np.linalg.norm(current_position_ - end)
            if is_collision or current_distance > allowed_distance:
                is_free = False
                break
        if is_free:
            free_length = segment_length
            collision_length = 0.0
        else:
            free_length = 0.0
            collision_length = segment_length
        return is_start_valid, free_length, collision_length
