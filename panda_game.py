import numpy as np
import multiprocessing
import queue
import time

from log_utils import print_and_log
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
        PandaGame.add_obstacles(config, panda_scene_manager)
        return panda_scene_manager

    @staticmethod
    def add_obstacles(config, panda_scene_manager):
        params_file = AbstractMotionPlanningGame.get_params_from_config(config)
        if 'no_obs' in params_file:
            obstacles_definitions_list = []
        else:
            with open(params_file, 'r') as f:
                obstacles_definitions_list = f.readlines()
        panda_scene_manager.add_obstacles(obstacles_definitions_list)

    @staticmethod
    def _truncate_virtual_state(state):
        truncated_state = np.maximum(np.minimum(state, 1.), -1.)
        truncated_distance = np.linalg.norm(state - truncated_state)
        return truncated_state, truncated_distance

    def is_free_state(self, state):
        return self._is_free_state_in_manager(state, self.panda_scene_manager)

    @staticmethod
    def _is_free_state_in_manager(state, panda_scene_manager):
        if any(np.abs(state) > 1.0):
            return False
        state_ = PandaGame._virtual_to_real_state(state, panda_scene_manager)
        panda_scene_manager.change_robot_joints(state_)
        is_collision = panda_scene_manager.simulation_step()[1]
        if not panda_scene_manager.is_close(state_):
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
        lower = np.array(self.panda_scene_manager.joints_lower_bounds)
        upper = np.array(self.panda_scene_manager.joints_upper_bounds)

        g = self._real_to_virtual_state(upper, self.panda_scene_manager)
        assert self.is_free_state(g)

        joints = 0.5 * upper + 0.5 * lower
        s = self._real_to_virtual_state(joints, self.panda_scene_manager)
        assert self.is_free_state(s)

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
            except queue.Empty:
                time.sleep(1.0)

    def check_terminal_segment(self, start, end):
        self.panda_scene_manager.reset_simulation()
        PandaGame.add_obstacles(self.config, self.panda_scene_manager)
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
        is_start_valid = self.panda_scene_manager.is_close(start) and not self.panda_scene_manager.is_collision()
        if not is_start_valid:
            # if the segment is not free
            return is_start_valid, 0.0, segment_length
        is_free = True
        self.panda_scene_manager.set_movement_target(end)
        steps = 0
        while not (self.panda_scene_manager.is_close(end) and not self.panda_scene_manager.is_moving()):
            is_collision = self.panda_scene_manager.simulation_step()[1]
            steps += 1
            if is_collision:
                is_free = False
                break
            if steps == 50000:
                current_joints, current_speed = self.panda_scene_manager.get_robot_state()
                distance_to_target = np.linalg.norm(np.array(current_joints) - np.array(end))
                speed = np.linalg.norm(np.array(current_speed))
                print_and_log('segment took too long, aborting for collision. distance to target {}, speed {}'.format(
                    distance_to_target, speed
                ))
                print_and_log('start {}'.format(start.tolist()))
                print_and_log('end {}'.format(end.tolist()))
                print_and_log('current {}'.format(np.array(current_joints).tolist()))
                print_and_log('')
                is_free = False
                break
        if is_free:
            free_length = segment_length
            collision_length = 0.0
        else:
            free_length = 0.0
            collision_length = segment_length
        return is_start_valid, free_length, collision_length
