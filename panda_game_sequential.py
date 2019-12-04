import os
from random import Random
import numpy as np
import multiprocessing
import queue
import time

from abstract_motion_planning_game_sequential import AbstractMotionPlanningGameSequential
from panda_scene_manager import PandaSceneManager


class PandaGameSequential(AbstractMotionPlanningGameSequential):
    def __init__(self, config):
        self.panda_scene_manager = PandaSceneManager.get_scene_manager(config)
        AbstractMotionPlanningGameSequential.__init__(self, config)

        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()

        self.number_of_workers = self._get_number_of_workers(config)

        self.worker_specific_requests_queue = [multiprocessing.Queue() for _ in range(self.number_of_workers)]
        self.worker_specific_response_queue = [multiprocessing.Queue() for _ in range(self.number_of_workers)]

        self.workers = [
            GameWorker(config, self.requests_queue, self.worker_specific_requests_queue[i], self.results_queue,
                       self.worker_specific_response_queue[i])
            for i in range(self.number_of_workers)
        ]

        for w in self.workers:
            w.start()

    def get_sizes(self):
        j = self.panda_scene_manager.number_of_joints
        return j * 2, j

    @staticmethod
    def _get_number_of_workers(config):
        # return 1  # for debug: set one worker
        # get configuration values
        max_episodes = max(config['general']['test_episodes'], config['general']['train_episodes_per_cycle'])
        # get available cpus
        cores = multiprocessing.cpu_count()
        max_cores_with_slack = max(cores - 2, 1)
        # the number of workers is the min between those
        return min(max_episodes, max_cores_with_slack)

    @staticmethod
    def _truncate_virtual_state(state):
        truncated_state = np.maximum(np.minimum(state, 1.), -1.)
        truncated_distance = np.linalg.norm(state - truncated_state)
        return truncated_state, truncated_distance

    @staticmethod
    def is_free_state_in_manager(state, panda_scene_manager):
        if any(np.abs(state) > 1.0):
            return False
        state_ = PandaGameSequential.virtual_to_real_state(state, panda_scene_manager)
        panda_scene_manager.change_robot_joints(state_)
        is_collision = panda_scene_manager.simulation_step()[1]
        if not panda_scene_manager.is_close(state_):
            return False
        if is_collision:
            return False
        return True

    def get_fixed_start_goal_pairs(self):
        # for now a long straight movement
        lower = np.array(self.panda_scene_manager.joints_lower_bounds).copy()
        upper = np.array(self.panda_scene_manager.joints_upper_bounds).copy()

        scenario = self.config['general']['scenario']
        if scenario == 'panda_no_obs':
            s = self.real_to_virtual_state(0.5 * upper + 0.5 * lower, self.panda_scene_manager)
            g = self.real_to_virtual_state(upper.copy(), self.panda_scene_manager)
        elif scenario == 'panda_easy':
            s = upper.copy()
            s[1] += lower[1]
            s[1] *= 0.5
            s[3] += lower[3]
            s[3] *= 0.5
            s = self.real_to_virtual_state(s, self.panda_scene_manager)

            g = upper.copy()
            g[0] = 0.7 * upper[0] + 0.3 * lower[0]
            g[2] = 0.1 * upper[2] + 0.9 * lower[2]
            g[3] = 0.4 * upper[3] + 0.6 * lower[3]
            g[4] = 0.7 * upper[4] + 0.3 * lower[4]
            g = self.real_to_virtual_state(g, self.panda_scene_manager)
        elif scenario == 'panda_hard':
            s = upper.copy()
            s[0] = 0.7 * upper[0] + 0.3 * lower[0]
            s[1] = 0.7 * upper[1] + 0.3 * lower[1]
            s[2] = 0.1 * upper[2] + 0.9 * lower[2]
            s[3] = 0.4 * upper[3] + 0.6 * lower[3]
            s[4] = 0.7 * upper[4] + 0.3 * lower[4]
            s[5] = 0.7 * upper[5] + 0.3 * lower[5]
            s[6] = 0.7 * upper[6] + 0.3 * lower[6]
            s = self.real_to_virtual_state(s, self.panda_scene_manager)

            g = upper.copy()
            g[0] = 0.1 * upper[0] + 0.9 * lower[0]
            g[1] = 0.6 * upper[1] + 0.4 * lower[1]
            g[3] = 0.1 * upper[3] + 0.9 * lower[3]
            g[4] = 0.4 * upper[4] + 0.6 * lower[4]
            g = self.real_to_virtual_state(g, self.panda_scene_manager)
        else:
            assert False
        # make sure states are free here
        assert self.is_free_state_in_manager(s, self.panda_scene_manager)
        assert self.is_free_state_in_manager(g, self.panda_scene_manager)
        # add the velocity dimensions
        state_size = self.panda_scene_manager.number_of_joints
        s = np.concatenate((s, np.array([0.] * state_size)), axis=0)
        g = np.concatenate((g, np.array([0.] * state_size)), axis=0)
        return [(np.array(s), np.array(g))]

    def run_episodes(self, start_goal_pairs, is_train, policy_function):
        # remaining_start_goal_pairs contains all the episodes not yet processed
        remaining_start_goal_pairs = {
            path_id: (start_goal_pair[0].copy(), start_goal_pair[1].copy())
            for path_id, start_goal_pair in enumerate(start_goal_pairs)
        }
        # active_start_goal_pairs contains all the episodes being processed
        active_start_goal_pairs = {}
        # data structure for free workers, and the assignment of workers
        free_workers = [i for i in range(self.number_of_workers)]
        active_path_id_to_worker = {}
        # holds the results for each path id
        states = {path_id: [remaining_start_goal_pairs[path_id][0]] for path_id in remaining_start_goal_pairs}
        goals = {path_id: remaining_start_goal_pairs[path_id][1] for path_id in remaining_start_goal_pairs}
        actions = {path_id: [] for path_id in remaining_start_goal_pairs}
        successes = {path_id: False for path_id in remaining_start_goal_pairs}
        costs = {path_id: [] for path_id in remaining_start_goal_pairs}

        while len(remaining_start_goal_pairs) + len(active_start_goal_pairs) > 0:
            # employ all workers
            self._employ_workers(remaining_start_goal_pairs, active_start_goal_pairs, free_workers,
                                 active_path_id_to_worker)
            # predict and run action in all simulators
            self._predict_and_play_actions(active_start_goal_pairs, states, goals, actions, policy_function, is_train,
                                           active_path_id_to_worker)

            self._process_action_results(active_start_goal_pairs, active_path_id_to_worker, free_workers, states, goals,
                                         successes, costs)

        # collect the results and return
        results = {
            path_id: (states[path_id], goals[path_id], actions[path_id], costs[path_id], successes[path_id])
            for path_id in states
        }
        return results

    def _employ_workers(self, remaining_start_goal_pairs, active_start_goal_pairs, free_workers,
                        active_path_id_to_worker):
        wait_for_episode_init = []
        while len(remaining_start_goal_pairs) > 0 and len(active_start_goal_pairs) < self.number_of_workers:
            # take a single path definition
            path_id = list(remaining_start_goal_pairs.keys())[0]
            start_goal = remaining_start_goal_pairs.pop(path_id)
            # mark as active
            assert path_id not in active_start_goal_pairs
            active_start_goal_pairs[path_id] = start_goal
            # assign to a worker
            worker_id = free_workers.pop()
            active_path_id_to_worker[path_id] = worker_id
            # initiate the episode within the worker
            message = (0, start_goal[0])
            self.worker_specific_requests_queue[worker_id].put(message)
            # mark the worker as "waiting to initialize"
            wait_for_episode_init.append(worker_id)
        for worker_id in wait_for_episode_init:
            self.worker_specific_response_queue[worker_id].get(block=True)

    def _predict_and_play_actions(self, active_start_goal_pairs, states, goals, actions, policy_function, is_train,
                                  active_path_id_to_worker):
        # fix all the active path ids
        path_ids = list(active_start_goal_pairs.keys())
        # generate the input for the policy function
        current_states = [states[path_id][-1] for path_id in path_ids]
        goal_states = [goals[path_id] for path_id in path_ids]
        # do action prediction
        predicted_actions = policy_function(current_states, goal_states, is_train)
        # play the action in the corresponding worker
        for i, path_id in enumerate(path_ids):
            action = predicted_actions[i]
            message = (1, action)
            worker_id = active_path_id_to_worker[path_id]
            self.worker_specific_requests_queue[worker_id].put(message)
            # also remember the action for the result
            actions[path_id].append(action)

    def _process_action_results(self, active_start_goal_pairs, active_path_id_to_worker, free_workers, states, goals,
                                successes, costs):
        active_path_ids = list(active_start_goal_pairs.keys())
        for path_id in active_path_ids:
            # get the worker response
            worker_id = active_path_id_to_worker[path_id]
            worker_response = self.worker_specific_response_queue[worker_id].get(block=True)
            (joints, velocity), is_collision = worker_response
            joints = self.real_to_virtual_state(joints, self.panda_scene_manager)
            new_state = np.concatenate((joints, velocity), axis=0)
            # save the new state
            states[path_id].append(new_state)
            goal_joints = goals[path_id][:self.panda_scene_manager.number_of_joints]
            goal_velocity = goals[path_id][self.panda_scene_manager.number_of_joints:]
            close_to_goal = self.are_close(joints, goal_joints) and self.are_close(velocity, goal_velocity)
            # set the cost, and success status
            if is_collision:
                cost = self.config['cost']['collision_cost']
            elif close_to_goal:
                cost = -self.config['cost']['goal_reward']
                successes[path_id] = True
            else:
                cost = self.config['cost']['free_cost']
            costs[path_id].append(cost)
            # episode terminated, remove from active pairs
            if is_collision or close_to_goal:
                active_start_goal_pairs.pop(path_id)
                worker_id = active_path_id_to_worker.pop(path_id)
                free_workers.append(worker_id)

    def get_free_start_goals(self, number_of_episodes):
        # put all requests
        for i in range(number_of_episodes):
            self.requests_queue.put((i, None))

        # pull all responses
        results = []
        for _ in range(number_of_episodes):
            response = self.results_queue.get(block=True)
            results.append(response)
        return results

    @staticmethod
    def virtual_to_real_state(s, panda_scene_manager):
        lower = np.array(panda_scene_manager.joints_lower_bounds)
        upper = np.array(panda_scene_manager.joints_upper_bounds)
        s_ = np.array(s)
        s_ = (upper - lower) * s_ + (upper + lower)
        s_ = s_ / 2.
        return s_

    @staticmethod
    def real_to_virtual_state(s, panda_scene_manager):
        lower = np.array(panda_scene_manager.joints_lower_bounds)
        upper = np.array(panda_scene_manager.joints_upper_bounds)
        denom = upper - lower
        s_ = np.array(s)
        s_ = 2 * s_ - (upper + lower)
        s_ = s_ / denom
        return s_

    @staticmethod
    def are_close(s1, s2):
        return np.linalg.norm(np.array(s1) - np.array(s2)) < 0.01


class GameWorker(multiprocessing.Process):
    def __init__(
            self, config, requests_queue, worker_specific_request_queue, results_queue, worker_specific_response_queue):
        multiprocessing.Process.__init__(self)
        self.config = config
        self.requests_queue = requests_queue
        self.worker_specific_request_queue = worker_specific_request_queue
        self.results_queue = results_queue
        self.worker_specific_response_queue = worker_specific_response_queue

        self._panda_scene_manager = None
        self._random = None
        self._is_in_episode = False

    def run(self):
        self._random = Random(os.getpid())
        self._panda_scene_manager = PandaSceneManager.get_scene_manager(self.config)
        while True:
            try:
                request = self.requests_queue.get(block=True, timeout=0.001)
                # for now there is only one message - the random start goal.
                self.results_queue.put(self.get_valid_start_goal())
            except queue.Empty:
                time.sleep(0.001)
            try:
                request = self.worker_specific_request_queue.get(block=True, timeout=0.001)
                message_type, message_params = request
                if message_type == 0:
                    # set the start position of the robot in the simulation
                    response = self.set_start_position(message_params)
                elif message_type == 1:
                    # apply a single action
                    response = self.apply_action(message_params)
                else:
                    assert False
                self.worker_specific_response_queue.put(response)
            except queue.Empty:
                time.sleep(1.0)

    def set_start_position(self, start_state):
        joints = self._panda_scene_manager.number_of_joints
        start = start_state[0:joints]
        velocity = start_state[joints:]
        assert all([v == 0. for v in velocity])
        start_ = PandaGameSequential.virtual_to_real_state(start, self._panda_scene_manager)
        self._panda_scene_manager.change_robot_joints(start_)
        is_collision = self._panda_scene_manager.simulation_step()[1]
        assert not is_collision
        self._is_in_episode = True
        return is_collision

    def apply_action(self, action):
        assert self._is_in_episode
        self._panda_scene_manager.set_movement_target(action)
        return self._panda_scene_manager.simulation_step()

    def get_valid_start_goal(self):
        self._is_in_episode = False
        while True:
            state_size = self._panda_scene_manager.number_of_joints
            virtual_state1 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            virtual_state2 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            if PandaGameSequential.are_close(virtual_state1, virtual_state2):
                continue
            if not PandaGameSequential.is_free_state_in_manager(virtual_state1, self._panda_scene_manager):
                continue
            if not PandaGameSequential.is_free_state_in_manager(virtual_state2, self._panda_scene_manager):
                continue
            virtual_state1 = virtual_state1 + [0.] * state_size
            virtual_state2 = virtual_state2 + [0.] * state_size
            return np.array(virtual_state1), np.array(virtual_state2)
