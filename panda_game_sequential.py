import os
import random
from random import Random
import numpy as np
import multiprocessing
import queue
import time

from abstract_motion_planning_game_sequential import AbstractMotionPlanningGameSequential
from panda_scene_manager import PandaSceneManager
from path_helper import get_start_goal_from_scenario


class PandaGameSequential(AbstractMotionPlanningGameSequential):
    def __init__(self, scenario, goal_reached_reward, collision_cost, keep_alive_cost, max_cores=None, max_steps=None,
                 goal_closeness_distance=0.01, max_queries_buffer_size=1000000, queries_update_freq=2):
        self.scenario = scenario
        self.max_steps = max_steps
        self.goal_reached_reward = goal_reached_reward
        self.collision_cost = collision_cost
        self.keep_alive_cost = keep_alive_cost

        self.panda_scene_manager = PandaSceneManager.get_scene_manager(scenario)
        self.state_size = self.get_state_space_size()
        self.action_size = self.get_action_space_size()
        self.closeness = goal_closeness_distance

        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()

        self._number_of_workers = self._get_number_of_workers(max_cores)

        self.worker_specific_requests_queue = [multiprocessing.Queue() for _ in range(self._number_of_workers)]
        self.worker_specific_response_queue = [multiprocessing.Queue() for _ in range(self._number_of_workers)]

        self.workers = [
            GameWorker(scenario, self.requests_queue, self.worker_specific_requests_queue[i], self.results_queue,
                       self.worker_specific_response_queue[i], self.closeness)
            for i in range(self._number_of_workers)
        ]

        self._max_queries_buffer_size = max_queries_buffer_size
        self._queries_update_freq = queries_update_freq
        self._queries_buffer = []
        self._queries_update_counter = 0

        for w in self.workers:
            w.start()

    def get_state_space_size(self):
        return self.panda_scene_manager.number_of_joints * 2

    def get_action_space_size(self):
        return self.panda_scene_manager.number_of_joints

    @staticmethod
    def _get_number_of_workers(max_cores):
        # return 1  # for debug: set one worker
        # get available cpus
        cores = multiprocessing.cpu_count()
        max_cores_with_slack = max(cores - 2, 1)
        if max_cores is not None:
            # the number of workers is the min between those
            return min(max_cores, max_cores_with_slack)
        else:
            return max_cores_with_slack

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
        with open(get_start_goal_from_scenario(self.scenario), 'r') as f:
            lines = [l.replace(os.linesep, '').replace('[', '').replace(']', '') for l in f.readlines()]

        result = []
        while len(result) * 2 < len(lines):
            index = len(result)
            start = np.array([float(f) for f in lines[2 * index].split(', ')])
            goal = np.array([float(f) for f in lines[2 * index + 1].split(', ')])
            # add velocity zero.
            start = np.concatenate((start, np.array([0.] * len(start))), axis=0)
            goal = np.concatenate((goal, np.array([0.] * len(goal))), axis=0)
            # append to results
            result.append((start, goal))
        return result

    def run_episodes(self, start_goal_pairs, is_train, policy_function):
        # remaining_start_goal_pairs contains all the episodes not yet processed
        remaining_start_goal_pairs = {
            path_id: (start_goal_pair[0].copy(), start_goal_pair[1].copy())
            for path_id, start_goal_pair in enumerate(start_goal_pairs)
        }
        # active_start_goal_pairs contains all the episodes being processed
        active_start_goal_pairs = {}
        # data structure for free workers, and the assignment of workers
        free_workers = [i for i in range(self._number_of_workers)]
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
        while len(remaining_start_goal_pairs) > 0 and len(active_start_goal_pairs) < self._number_of_workers:
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
            close_to_goal = self.are_close(joints, goal_joints, self.closeness) and self.are_close(
                velocity, goal_velocity, self.closeness)
            # set the cost, and success status
            if is_collision:
                cost = self.collision_cost
            elif close_to_goal:
                cost = -self.goal_reached_reward
                successes[path_id] = True
            else:
                cost = self.keep_alive_cost
            costs[path_id].append(cost)
            max_steps_reached = False
            if self.max_steps is not None:
                max_steps_reached = len(states[path_id]) == self.max_steps + 1
            # episode terminated, remove from active pairs
            if is_collision or close_to_goal or max_steps_reached:
                active_start_goal_pairs.pop(path_id)
                worker_id = active_path_id_to_worker.pop(path_id)
                free_workers.append(worker_id)

    def get_free_start_goals(self, number_of_episodes, curriculum_coefficient):
        # how many episodes to collect?
        if len(self._queries_buffer) < number_of_episodes:
            # initial steps - collect enough to sample randomly for a couple of cycles
            queries_to_generate = max(number_of_episodes * self._queries_update_freq, self._number_of_workers)
        else:
            if self._queries_update_counter == self._queries_update_freq:
                queries_to_generate = self._number_of_workers
                self._queries_update_counter = 0
            else:
                queries_to_generate = 0
                self._queries_update_counter += 1
        # collect
        new_queries = self._get_free_start_goals_from_game(queries_to_generate, curriculum_coefficient)
        self._queries_buffer.extend(new_queries)
        # shuffle and remove extra
        random.shuffle(self._queries_buffer)
        self._queries_buffer = self._queries_buffer[:self._max_queries_buffer_size]
        return self._queries_buffer[:number_of_episodes]

    def _get_free_start_goals_from_game(self, number_of_episodes, curriculum_coefficient):
        # put all requests
        for i in range(number_of_episodes):
            self.requests_queue.put((i, curriculum_coefficient))

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
    def are_close(s1, s2, closeness):
        return np.linalg.norm(np.array(s1) - np.array(s2)) < closeness


class GameWorker(multiprocessing.Process):
    def __init__(self, scenario, requests_queue, worker_specific_request_queue, results_queue,
                 worker_specific_response_queue, closeness):
        multiprocessing.Process.__init__(self)
        self.scenario = scenario
        self.requests_queue = requests_queue
        self.worker_specific_request_queue = worker_specific_request_queue
        self.results_queue = results_queue
        self.worker_specific_response_queue = worker_specific_response_queue
        self.closeness = closeness

        self._panda_scene_manager = None
        self._random = None
        self._is_in_episode = False

    def run(self):
        self._random = Random(os.getpid())
        self._panda_scene_manager = PandaSceneManager.get_scene_manager(self.scenario)
        while True:
            try:
                i, curriculum_coefficient = self.requests_queue.get(block=True, timeout=0.001)
                # for now there is only one message - the random start goal.
                self.results_queue.put(self.get_valid_start_goal(curriculum_coefficient))
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
        current_joints, _ = self._panda_scene_manager.get_robot_state()
        current_virtual_joints = PandaGameSequential.real_to_virtual_state(current_joints, self._panda_scene_manager)
        new_virtual_action = current_virtual_joints + action
        new_virtual_action = np.maximum(new_virtual_action, -1.)
        new_virtual_action = np.minimum(new_virtual_action, 1.)
        movement_target = PandaGameSequential.virtual_to_real_state(new_virtual_action, self._panda_scene_manager)
        self._panda_scene_manager.set_movement_target(movement_target)
        return self._panda_scene_manager.simulation_step()

    def get_valid_start_goal(self, curriculum_coefficient):
        self._is_in_episode = False
        while True:
            state_size = self._panda_scene_manager.number_of_joints
            virtual_state1 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            if not PandaGameSequential.is_free_state_in_manager(virtual_state1, self._panda_scene_manager):
                continue
            virtual_state2 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            if curriculum_coefficient is None:
                # do not use curriculum
                if PandaGameSequential.are_close(virtual_state1, virtual_state2, self.closeness):
                    continue
            else:
                # use curriculum
                direction = virtual_state2.copy()
                direction = direction / np.linalg.norm(direction)
                direction *= self.closeness
                size = np.random.uniform(1., curriculum_coefficient)
                direction *= size
                virtual_state2 = virtual_state1 + direction

            if not PandaGameSequential.is_free_state_in_manager(virtual_state2, self._panda_scene_manager):
                continue

            virtual_state1 = np.concatenate((np.array(virtual_state1), np.array([0.] * state_size)), axis=0)
            virtual_state2 = np.concatenate((np.array(virtual_state2), np.array([0.] * state_size)), axis=0)
            return np.array(virtual_state1), np.array(virtual_state2)
