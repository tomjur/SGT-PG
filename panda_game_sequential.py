import os
import random
from random import Random
import numpy as np
import multiprocessing
import queue
import time
import tensorflow as tf

from abstract_motion_planning_game_sequential import AbstractMotionPlanningGameSequential
from network_sequential import NetworkSequential
from panda_scene_manager import PandaSceneManager
from path_helper import get_start_goal_from_scenario


class PandaGameSequential(AbstractMotionPlanningGameSequential):
    def __init__(self, config, max_cores=None):
        self.scenario = config['general']['scenario']

        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()

        self._number_of_workers = self._get_number_of_workers(max_cores)

        self.worker_specific_requests_queue = [multiprocessing.Queue() for _ in range(self._number_of_workers)]
        self.worker_specific_response_queue = [multiprocessing.Queue() for _ in range(self._number_of_workers)]

        self.workers = [
            GameWorker(config, self.requests_queue, self.worker_specific_requests_queue[i], self.results_queue,
                       self.worker_specific_response_queue[i], self._number_of_workers)
            for i in range(self._number_of_workers)
        ]

        for w in self.workers:
            w.start()

    def get_state_space_size(self):
        return 9

    def get_action_space_size(self):
        return 9

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

    def get_number_of_workers(self):
        return self._number_of_workers

    def get_fixed_start_goal_pairs(self):
        with open(get_start_goal_from_scenario(self.scenario), 'r') as f:
            lines = [l.replace(os.linesep, '').replace('[', '').replace(']', '') for l in f.readlines()]

        result = []
        while len(result) * 2 < len(lines):
            index = len(result)
            goal = np.array([float(f) for f in lines[2 * index + 1].split(', ')])
            if '_fixed_start' in self.scenario:
                start = np.array([0.0 for _ in range(len(goal))])
            else:
                start = np.array([float(f) for f in lines[2 * index].split(', ')])
            # append to results
            result.append((start, goal))
        return result

    def run_episodes(self, start_goal_pairs, is_train):
        for path_id, (start, goal) in enumerate(start_goal_pairs):
            # message 1 is "play episode"
            message = (1, (path_id, start, goal, is_train))
            self.requests_queue.put(message)

        result = {}
        for i in range(len(start_goal_pairs)):
            path_id, states, actions, costs, goal, is_successful = self.results_queue.get(block=True)
            result[path_id] = (states, goal, actions, costs, is_successful)
            if i % 10 == 9:
                print('finished {} episodes...'.format(i+1))
        return result

    def get_free_start_goals(self, number_of_episodes, curriculum_coefficient):
        # collect
        print('generating {} new queries'.format(number_of_episodes))
        new_queries = self._get_free_start_goals_from_game(number_of_episodes, curriculum_coefficient)
        distances = np.mean([np.linalg.norm(g-s) for s, g in new_queries])
        print('done generating queries, start->goal mean distance is {}'.format(distances))
        return new_queries

    def _get_free_start_goals_from_game(self, number_of_episodes, curriculum_coefficient):
        # put all requests
        for _ in range(number_of_episodes):
            # message 0 is "generate start-goal"
            self.requests_queue.put((0, curriculum_coefficient))

        # pull all responses
        results = []
        for _ in range(number_of_episodes):
            response = self.results_queue.get(block=True)
            results.append(response)
        return results

    def update_weights(self, new_weights):
        for request_queue in self.worker_specific_requests_queue:
            message = (0, new_weights)
            request_queue.put(message)

        for response_queue in self.worker_specific_response_queue:
            response = response_queue.get(block=True)


class GameWorker(multiprocessing.Process):
    def __init__(self, config, requests_queue, worker_specific_request_queue, results_queue,
                 worker_specific_response_queue, number_of_workers):
        multiprocessing.Process.__init__(self)
        self.config = config

        self.goal_reached_reward = config['cost']['goal_reward']
        self.collision_cost = config['cost']['collision_cost']
        self.keep_alive_cost = config['cost']['free_cost']
        self.closeness = config['panda_game']['goal_closeness_distance']
        self.limit_action_distance = config['panda_game']['limit_action_distance']
        self.max_steps = config['panda_game']['max_steps']
        self.add_distance_to_failed = config['panda_game']['add_distance_to_failed']
        self.add_distance_to_keep_alive = config['panda_game']['add_distance_to_keep_alive']
        self.add_distance_to_success = config['panda_game']['add_distance_to_success']

        self.requests_queue = requests_queue
        self.worker_specific_request_queue = worker_specific_request_queue
        self.results_queue = results_queue
        self.worker_specific_response_queue = worker_specific_response_queue
        self.number_of_workers = number_of_workers

        self._random = None
        self._panda_scene_manager = None
        self._network = None

    def run(self):
        my_pid = os.getpid()
        self._random = Random(my_pid)
        self._panda_scene_manager = PandaSceneManager.get_scene_manager(self.config['general']['scenario'])
        number_of_joints = self._panda_scene_manager.number_of_joints

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(
                    per_process_gpu_memory_fraction=self.config['general']['actor_gpu_usage'])
        )) as sess:

            self._network = NetworkSequential(self.config, number_of_joints, number_of_joints, is_rollout_agent=False)

            short_sleep = 0.001
            long_sleep = 1.
            while True:
                try:
                    message_type, message_params = self.requests_queue.get(block=True, timeout=short_sleep)
                    # check message type:
                    if message_type == 0:
                        # get free start goal
                        curriculum_coefficient = message_params
                        self.results_queue.put(self.get_valid_start_goal(curriculum_coefficient))
                    elif message_type == 1:
                        # play episode
                        path_id, start, goal, is_train = message_params
                        states, actions, costs, is_successful = self.play_episode(start, goal, sess, is_train)
                        response_tuple = (
                            path_id, states, actions, costs, goal, is_successful
                        )
                        self.results_queue.put(response_tuple)
                    else:
                        assert False

                except queue.Empty:
                    time.sleep(short_sleep)
                try:
                    request = self.worker_specific_request_queue.get(block=True, timeout=short_sleep)
                    message_type, message_params = request
                    if message_type == 0:
                        new_weights = message_params
                        self.update_network_weights(new_weights, sess)
                        self.worker_specific_response_queue.put(my_pid)
                    else:
                        assert False

                except queue.Empty:
                    time.sleep(long_sleep)

    def update_network_weights(self, new_weights, sess):
        self._network.set_policy_weights(sess, new_weights)

    def play_episode(self, start, goal, sess, is_train):
        self._set_start_position(start)
        states = [start]
        actions = []
        costs = []
        is_successful = False

        goals = [goal]
        goal_joints = goal[:self._panda_scene_manager.number_of_joints]

        should_stop = False
        distance_covered = 0.0
        counter = 0
        while not should_stop:
            # predict action
            action = self._network.predict_policy([states[-1]], goals, sess, is_train)[0]
            # add as-is to the list of actions
            actions.append(action)
            # execute, get and save the new state
            joints, is_collision = self._apply_action(action)
            joints = self._real_to_virtual_state(joints)
            new_state = np.array(joints)
            new_distance = np.linalg.norm(new_state - states[-1])
            distance_covered += new_distance
            states.append(new_state)

            # compute the costs
            # if the agent is close to the goal
            close_to_goal = self._are_close(joints, goal_joints, self.closeness)
            distance_to_goal = np.linalg.norm(new_state - goal_joints)

            # if the agent moved too much already
            distance_limit_reached = distance_covered > 2 * np.sqrt(self._panda_scene_manager.number_of_joints) * 10
            # if the agent did not move (stuck in place)
            stationary_agent = new_distance < self.closeness / 10.
            # max steps reached
            max_counter = counter >= self.max_steps

            if is_collision:
                cost = self.collision_cost * (1. + self.add_distance_to_failed * distance_to_goal)
                should_stop = True
            elif close_to_goal:
                cost = -self.goal_reached_reward * (1. + self.add_distance_to_success * distance_to_goal)
                is_successful = True
                should_stop = True
            elif distance_limit_reached or stationary_agent or max_counter:
                cost = self.collision_cost * (1. + self.add_distance_to_failed * distance_to_goal)
                should_stop = True
            else:
                cost = self.keep_alive_cost * (1. + self.add_distance_to_keep_alive * distance_to_goal)
            costs.append(cost)
            counter += 1

        return states, actions, costs, is_successful

    def _set_start_position(self, start_state):
        joints = self._panda_scene_manager.number_of_joints
        start = start_state[0:joints]
        velocity = start_state[joints:]
        assert all([v == 0. for v in velocity])
        start_ = self._virtual_to_real_state(start)
        self._panda_scene_manager.change_robot_joints(start_)
        is_collision = self._panda_scene_manager.simulation_step()[1]
        assert not is_collision

    def _limit_action(self, action):
        if self.limit_action_distance is not None:
            assert self.limit_action_distance > 0.
            current_norm = np.linalg.norm(action)
            if current_norm > self.limit_action_distance:
                action = (self.limit_action_distance / current_norm) * action
        return action

    def _apply_action(self, action):
        # in the environment execute a limited action only!
        action = self._limit_action(action)
        current_joints, _ = self._panda_scene_manager.get_robot_state()
        current_virtual_joints = self._real_to_virtual_state(current_joints)
        new_virtual_action = current_virtual_joints + action
        new_virtual_action = np.maximum(new_virtual_action, -1.)
        new_virtual_action = np.minimum(new_virtual_action, 1.)
        movement_target = self._virtual_to_real_state(new_virtual_action)
        if self.config['panda_game']['move'] == 'single-action':
            # only take a single action towards the goal
            self._panda_scene_manager.set_movement_target(movement_target)
            self._panda_scene_manager.simulation_step()
        elif self.config['panda_game']['move'] == 'multi-action-smooth':
            # execute the smooth motion controller
            self._panda_scene_manager.smooth_walk(movement_target, max_target_distance=1., sensitivity=0.01)

        else:
            # move option undefined
            assert False
        is_collision = self._panda_scene_manager.is_collision()
        joints, _ = self._panda_scene_manager.get_robot_state()
        if not is_collision:
            # set velocity in simulation to zero
            self._panda_scene_manager.change_robot_joints(joints)
            new_joints = self._panda_scene_manager.get_robot_state()[0]
            assert self._are_close(joints, new_joints, self.closeness)
            assert not self._panda_scene_manager.is_collision()
            joints = new_joints
        return joints, is_collision

    def get_valid_start_goal(self, curriculum_coefficient):
        while True:
            state_size = self._panda_scene_manager.number_of_joints
            if '_fixed_start' in self.config['general']['scenario']:
                virtual_state1 = [0.0 for _ in range(state_size)]
            else:
                virtual_state1 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            if not self._is_free_state(virtual_state1):
                continue
            virtual_state2 = [self._random.uniform(-1., 1.) for _ in range(state_size)]
            if curriculum_coefficient is None:
                # do not use curriculum
                if self._are_close(virtual_state1, virtual_state2, self.closeness):
                    continue
            else:
                # use curriculum
                curriculum_coefficient = min(curriculum_coefficient, 6. / self.closeness)
                direction = virtual_state2.copy()
                direction = direction / np.linalg.norm(direction)
                direction *= self.closeness
                size = self._random.uniform(1., curriculum_coefficient)
                direction *= size
                virtual_state2 = virtual_state1 + direction

            if not self._is_free_state(virtual_state2):
                continue

            return np.array(virtual_state1), np.array(virtual_state2)

    @staticmethod
    def _truncate_virtual_state(state):
        truncated_state = np.maximum(np.minimum(state, 1.), -1.)
        truncated_distance = np.linalg.norm(state - truncated_state)
        return truncated_state, truncated_distance

    def _is_free_state(self, state):
        if any(np.abs(state) > 1.0):
            return False
        state_ = self._virtual_to_real_state(state)
        self._panda_scene_manager.change_robot_joints(state_)
        is_collision = self._panda_scene_manager.simulation_step()[1]
        if not self._panda_scene_manager.is_close(state_):
            return False
        if is_collision:
            return False
        return True

    def _virtual_to_real_state(self, virtual_state):
        lower = np.array(self._panda_scene_manager.joints_lower_bounds)
        upper = np.array(self._panda_scene_manager.joints_upper_bounds)
        s_ = np.array(virtual_state)
        s_ = (upper - lower) * s_ + (upper + lower)
        s_ = s_ / 2.
        return s_

    def _real_to_virtual_state(self, real_state):
        lower = np.array(self._panda_scene_manager.joints_lower_bounds)
        upper = np.array(self._panda_scene_manager.joints_upper_bounds)
        denom = upper - lower
        s_ = np.array(real_state)
        s_ = 2 * s_ - (upper + lower)
        s_ = s_ / denom
        return s_

    @staticmethod
    def _are_close(s1, s2, closeness):
        return np.linalg.norm(np.array(s1) - np.array(s2)) < closeness