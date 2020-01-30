import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from gradient_limit_manager import GradientLimitManager
from network_utils import optimize_by_loss, get_activation


class Network:
    def __init__(self, config, game):
        self.config = config
        self.state_size = game.get_state_size()
        self.number_of_middle_states = self.config['model']['number_of_middle_states']

        self.start_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), name='start_inputs')
        self.goal_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), name='goal_inputs')
        self.next_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), name='next_inputs')
        self.label_inputs = tf.compat.v1.placeholder(tf.float32, (None, 1), name='label_inputs')

        self.gradient_limit_manager = GradientLimitManager(
            gradient_limit=config['policy']['gradient_limit'],
            gradient_limit_quantile=config['policy']['gradient_limit_quantile'],
            gradient_history_limit=config['policy']['gradient_history_limit'],
        )

        # network is comprised of several policies
        self.policy_network = PolicyNetwork(
                config, self.state_size, self.start_inputs, self.goal_inputs, self.next_inputs,
                self.label_inputs, self.gradient_limit_manager
            )
        self.value_networks = [
            ValueNetwork(
                config, self.state_size, self.start_inputs, self.goal_inputs, self.label_inputs, level
            )
            for level in range(self.number_of_middle_states)
        ]

        # this is the prediction over the entire subtree (element at index l contains the entire trajectory prediction
        # for a tree with l levels
        self.policy_tree_prediction = self._get_policy_tree(
            self.start_inputs, self.goal_inputs, self.number_of_middle_states, False)
        self.test_policy_tree_prediction = self._get_policy_tree(
            self.start_inputs, self.goal_inputs, self.number_of_middle_states, True)

    def _get_policy_tree(self, start_inputs, goal_inputs, horizon, take_mean):
        current_policy_distribution = self.policy_network.reuse_policy_network(
            start_inputs, goal_inputs)
        if take_mean:
            current_state = [current_policy_distribution.mean()]
        else:
            current_state = [current_policy_distribution.sample()]
        if horizon == 1:
            return current_state
        suffix_states = self._get_policy_tree(current_state[-1], goal_inputs, horizon-1, take_mean)
        return current_state + suffix_states

    def predict_policy(self, start_inputs, goal_inputs, sess, is_train):
        tree = self.policy_tree_prediction if is_train else self.test_policy_tree_prediction
        return sess.run(tree, self._generate_feed_dictionary(start_inputs, goal_inputs))

    def predict_value(self, current_inputs, goal_inputs, sess, level):
        return sess.run(
            self.value_networks[level].value_estimation, self._generate_feed_dictionary(current_inputs, goal_inputs)
        )

    def train_policy(self, start_inputs, goal_inputs, next_inputs, labels, sess, symmetric=False):
        network = self.policy_network
        feed_dictionary = self._generate_feed_dictionary(
            start_inputs, goal_inputs, next_inputs=next_inputs, labels=labels, symmetric=symmetric)
        self.gradient_limit_manager.update_feed_dict(feed_dictionary, network.name_prefix)
        ops = [
            network.initial_gradients_norm, network.clipped_gradients_norm, network.optimization_summaries,
            network.cost_loss, network.optimize
        ]
        result = sess.run(ops, feed_dictionary)
        initial_gradients = result[0]
        self.gradient_limit_manager.update_gradient_limit(network.name_prefix, initial_gradients)
        return result

    def train_value_estimation(self, current_inputs, goal_inputs, labels, sess, level):
        network = self.value_networks[level]
        feed_dictionary = self._generate_feed_dictionary(
            current_inputs, goal_inputs, labels=labels)
        network.gradient_limit_manager.update_feed_dict(feed_dictionary, network.name_prefix)
        ops = [
            network.initial_gradients_norm, network.clipped_gradients_norm, network.optimization_summaries,
            network.prediction_loss, network.optimize
        ]
        result = sess.run(ops, feed_dictionary)
        initial_gradients = result[0]
        network.gradient_limit_manager.update_gradient_limit(network.name_prefix, initial_gradients)
        return result[2:]

    def decrease_learn_rates(self, sess):
        return sess.run([self.policy_network.decrease_learn_rate_op])

    def decrease_base_std(self, sess):
        return sess.run(self.policy_network.decrease_base_std_op)

    def get_learn_rates(self, sess):
        return sess.run(self.policy_network.learn_rate_variable)

    def get_base_stds(self, sess):
        return sess.run(self.policy_network.base_std_variable)

    def update_baseline_policy(self, sess):
        sess.run(self.policy_network.assign_to_baseline_ops)

    def get_all_variables(self):
        vars = self.policy_network.model_variables
        for network in self.value_networks:
            vars = vars + network.model_variables
        return vars

    def _generate_feed_dictionary(self, start_inputs, goal_inputs, next_inputs=None, labels=None, symmetric=False):
        start_inputs_ = np.array(start_inputs)
        goal_inputs_ = np.array(goal_inputs)

        if next_inputs is None:
            next_inputs_ = None
        else:
            next_inputs_ = np.array(next_inputs)

        if labels is None:
            labels_ = None
        else:
            labels_ = np.array(labels)
            if len(labels_.shape) == 1:
                labels_ = np.expand_dims(labels_, axis=1)

        if symmetric:
            # make the buffer invariant to start goal direction
            temp_start = start_inputs_
            start_inputs_ = np.concatenate((start_inputs_, goal_inputs_), axis=0)
            goal_inputs_ = np.concatenate((goal_inputs_, temp_start), axis=0)
            if next_inputs_ is not None:
                next_inputs_ = np.concatenate((next_inputs_, next_inputs_), axis=0)
            if labels_ is not None:
                labels_ = np.concatenate((labels_, labels_), axis=0)

        feed_dictionary = {
            self.start_inputs: start_inputs_,
            self.goal_inputs: goal_inputs_,
        }

        if next_inputs_ is not None:
            feed_dictionary[self.next_inputs] = next_inputs_

        if labels_ is not None:
            feed_dictionary[self.label_inputs] = labels_

        return feed_dictionary


class PolicyNetwork:
    def __init__(self, config, state_size, start_inputs, goal_inputs, next_inputs, label_inputs,
                 gradient_limit_manager):
        self.config = config
        self.name_prefix = 'sequential_subgoal_policy'
        self.state_size = state_size
        self._reuse = False

        self.start_inputs = start_inputs
        self.goal_inputs = goal_inputs
        self.next_inputs = next_inputs
        self.label_inputs = label_inputs

        self.gradient_limit_placeholder = gradient_limit_manager.set_key(self.name_prefix)

        self.base_std_variable = tf.Variable(
            self.config['policy']['base_std'], trainable=False, name='base_std_variable')
        new_base_std = self.config['policy']['std_decrease_rate'] * self.base_std_variable
        self.decrease_base_std_op = tf.compat.v1.assign(self.base_std_variable, new_base_std)

        # get the prediction distribution
        self.prediction_distribution, self.model_variables = self._create_network(self.start_inputs, self.goal_inputs)

        # get the baseline prediction distribution (remains fixed during optimization)
        self.baseline_prediction_distribution, self.baseline_model_variables = self._create_network(
            self.start_inputs, self.goal_inputs, is_baseline=True)

        # to update baseline to the optimized:
        self.assign_to_baseline_ops = self.get_assignment_between_policies(
            self.model_variables, self.baseline_model_variables
        )

        # compute the policies ratio
        log_likelihood = self.prediction_distribution.log_prob(self.next_inputs)
        log_likelihood_baseline = self.baseline_prediction_distribution.log_prob(self.next_inputs)
        policy_ratio = tf.exp(log_likelihood - log_likelihood_baseline)

        # clip the ratio
        epsilon = self.config['policy']['ppo_epsilon']
        clipped_ratio = tf.maximum(tf.minimum(policy_ratio, 1. - epsilon), 1. + epsilon)

        # add advantage
        advantage = tf.expand_dims(policy_ratio, axis=1) * self.label_inputs
        clipped_advantage = tf.expand_dims(clipped_ratio, axis=1) * self.label_inputs

        # compute the cost loss
        self.cost_loss = tf.reduce_mean(tf.maximum(advantage, clipped_advantage))

        # compute the max entropy loss
        self.entropy_loss = 0.0
        if self.config['policy']['max_entropy_coefficient'] > 0. and self.config['policy']['learn_std']:
            entropy = self.prediction_distribution.entropy()
            self.entropy_loss = -self.config['policy']['max_entropy_coefficient'] * tf.reduce_mean(entropy)

        # optimize
        self.total_loss = self.cost_loss + self.entropy_loss
        self.learn_rate_variable = tf.Variable(
            self.config['policy']['learning_rate'], trainable=False, name='learn_rate_variable')
        new_learn_rate = tf.maximum(self.config['policy']['learning_rate_minimum'],
                                    self.config['policy']['learning_rate_decrease_rate'] * self.learn_rate_variable)
        self.decrease_learn_rate_op = tf.compat.v1.assign(self.learn_rate_variable, new_learn_rate)

        self.initial_gradients_norm, self.clipped_gradients_norm, self.optimize = \
            optimize_by_loss(
                self.total_loss, self.model_variables, self.learn_rate_variable, self.gradient_limit_placeholder
            )

        norm = [(v, np.product([int(s) for s in v.shape])) for v in self.model_variables]
        norm = [tf.reshape(t[0], (t[1],)) for t in norm]
        norm = tf.concat(norm, axis=-1)
        norm = tf.norm(norm)

        # summaries
        merge_summaries = [
            tf.compat.v1.summary.scalar('{}_cost_loss'.format(self.name_prefix), self.cost_loss),
            tf.compat.v1.summary.scalar('{}_entropy_loss'.format(self.name_prefix), self.entropy_loss),
            tf.compat.v1.summary.scalar('{}_total_loss'.format(self.name_prefix), self.total_loss),
            tf.compat.v1.summary.scalar('{}_learn_rate'.format(self.name_prefix), self.learn_rate_variable),
            tf.compat.v1.summary.scalar('{}_weights_norm'.format(self.name_prefix), norm),
        ]
        if self.initial_gradients_norm is not None:
            merge_summaries.append(
                tf.compat.v1.summary.scalar(
                    '{}_initial_gradients_norm'.format(self.name_prefix),
                    self.initial_gradients_norm
                )
            )
        if self.clipped_gradients_norm is not None:
            merge_summaries.append(
                tf.compat.v1.summary.scalar(
                    '{}_clipped_gradients_norm'.format(self.name_prefix),
                    self.clipped_gradients_norm
                )
            )
        self.optimization_summaries = tf.compat.v1.summary.merge(merge_summaries)

    @staticmethod
    def get_assignment_between_policies(source_policy_variables, target_policy_variables):
        assign_ops = [
            tf.compat.v1.assign(var, source_policy_variables[i])
            for i, var in enumerate(target_policy_variables)
        ]
        return assign_ops

    def reuse_policy_network(self, start_inputs, goal_inputs):
        policy_distribution, model_variables = self._create_network(start_inputs, goal_inputs)
        assert len(model_variables) == 0
        return policy_distribution

    def _create_network(self, start_inputs, goal_inputs, is_baseline=False):
        if is_baseline:
            name_prefix = '{}_baseline'.format(self.name_prefix)
            reuse = False
        else:
            name_prefix = self.name_prefix
            reuse = self._reuse
        variable_count = len(tf.compat.v1.trainable_variables())
        activation = get_activation(self.config['policy']['activation'])
        network_layers = self.config['policy']['layers']
        learn_std = self.config['policy']['learn_std']
        distance_adaptive_std = self.config['policy']['distance_adaptive_std']

        base_std = tf.squeeze(tf.tile(tf.reshape(self.base_std_variable, (1, 1)), (self.state_size, 1)), axis=-1)
        if distance_adaptive_std:
            # if the std is distance adaptive, the base std if of the form:
            # (base_std_from_config + softplus(learnable_parameters)) * dist(start, goal)
            learnable_distance_coeff = tf.layers.dense(
                tf.ones((1, 1)), self.state_size, activation=tf.nn.softplus, name='{}_std_coeff'.format(name_prefix),
                reuse=reuse, use_bias=False
            )

            # with tf.variable_scope("std_scope", reuse=reuse):
            #     learnable_distance_coeff = tf.nn.softplus(
            #         tf.Variable([0.0]*self.state_size, trainable=True, shape=self.state_size)
            #     )
            base_std = base_std + learnable_distance_coeff
            distance = tf.linalg.norm(start_inputs - goal_inputs, axis=1)
            base_std = tf.expand_dims(distance, axis=1) * base_std
        current_input = tf.concat((start_inputs, goal_inputs), axis=1)
        shift = (start_inputs + goal_inputs) * 0.5
        if self.config['policy']['include_middle_state_as_input']:
            current_input = tf.concat((current_input, shift), axis=1)

        current = current_input
        for i, layer_size in enumerate(network_layers):
            current = tf.layers.dense(
                current, layer_size, activation=activation, name='{}_layer_{}'.format(name_prefix, i), reuse=reuse,
            )

        if learn_std:
            normal_dist_parameters = tf.layers.dense(
                current, self.state_size * 2, activation=None,
                name='{}_normal_dist_parameters'.format(name_prefix), reuse=reuse,
            )
            split_normal_dist_parameters = tf.split(normal_dist_parameters, 2, axis=1)
            bias = split_normal_dist_parameters[0]
            std = split_normal_dist_parameters[1]
            std = tf.math.softplus(std)
            std = std + base_std
        else:
            normal_dist_parameters = tf.layers.dense(
                current,  self.state_size, activation=None,
                name='{}_normal_dist_parameters'.format(name_prefix), reuse=reuse,
            )
            bias = normal_dist_parameters
            std = base_std

        if self.config['policy']['bias_activation_is_tanh']:
            bias = tf.tanh(bias)

        if self.config['policy']['bias_around_midpoint']:
            bias = bias + shift

        distribution = tfp.distributions.MultivariateNormalDiag(loc=bias, scale_diag=std)
        model_variables = tf.compat.v1.trainable_variables()[variable_count:]
        if reuse:
            assert len(model_variables) == 0
        elif not is_baseline:
            self._reuse = True
        return distribution, model_variables


class ValueNetwork:
    def __init__(self, config, state_size, current_state_inputs, goal_inputs, label_inputs, level):
        self.config = config
        self.name_prefix = 'sequential_subgoal_value_estimator_level_{}'.format(level)
        self.state_size = state_size
        self.level = level
        self._reuse = False

        self.current_state_inputs = current_state_inputs
        self.goal_inputs = goal_inputs
        self.label_inputs = label_inputs

        self.gradient_limit_manager = GradientLimitManager(
            gradient_limit=config['value_estimator']['gradient_limit'],
            gradient_limit_quantile=config['value_estimator']['gradient_limit_quantile'],
            gradient_history_limit=config['value_estimator']['gradient_history_limit'],
        )
        self.gradient_limit_placeholder = self.gradient_limit_manager.set_key(self.name_prefix)

        # get the prediction
        self.value_estimation, self.model_variables = self._create_network(self.current_state_inputs, self.goal_inputs)

        # compute the prediction loss
        self.prediction_loss = tf.losses.mean_squared_error(self.label_inputs, self.value_estimation)

        # optimize
        self.learn_rate_variable = tf.Variable(
            self.config['value_estimator']['learning_rate'], trainable=False, name='learn_rate_variable')
        new_learn_rate = tf.maximum(
            self.config['value_estimator']['learning_rate_minimum'],
            self.config['value_estimator']['learning_rate_decrease_rate'] * self.learn_rate_variable
        )
        self.decrease_learn_rate_op = tf.compat.v1.assign(self.learn_rate_variable, new_learn_rate)

        self.initial_gradients_norm, self.clipped_gradients_norm, self.optimize = \
            optimize_by_loss(
                self.prediction_loss, self.model_variables, self.learn_rate_variable, self.gradient_limit_placeholder
            )

        # summaries
        merge_summaries = [
            tf.compat.v1.summary.scalar('{}_value_prediction_loss'.format(self.name_prefix), self.prediction_loss),
            tf.compat.v1.summary.scalar('{}_learn_rate'.format(self.name_prefix), self.learn_rate_variable),
        ]
        if self.initial_gradients_norm is not None:
            merge_summaries.append(
                tf.compat.v1.summary.scalar(
                    '{}_initial_gradients_norm'.format(self.name_prefix),
                    self.initial_gradients_norm
                )
            )
        if self.clipped_gradients_norm is not None:
            merge_summaries.append(
                tf.compat.v1.summary.scalar(
                    '{}_clipped_gradients_norm'.format(self.name_prefix),
                    self.clipped_gradients_norm
                )
            )
        self.optimization_summaries = tf.compat.v1.summary.merge(merge_summaries)

    def _create_network(self, current_state_inputs, goal_inputs):
        reuse = self._reuse
        name_prefix = self.name_prefix
        variable_count = len(tf.compat.v1.trainable_variables())
        activation = get_activation(self.config['value_estimator']['activation'])
        network_layers = self.config['value_estimator']['layers']

        current_input = tf.concat((current_state_inputs, goal_inputs), axis=1)

        current = current_input
        for i, layer_size in enumerate(network_layers):
            current = tf.layers.dense(
                current, layer_size, activation=activation, name='{}_layer_{}'.format(name_prefix, i), reuse=reuse,
            )

        value_estimation = tf.layers.dense(
            current, 1, activation=None, name='{}_prediction'.format(name_prefix), reuse=reuse
        )

        model_variables = tf.compat.v1.trainable_variables()[variable_count:]
        if reuse:
            assert len(model_variables) == 0
        else:
            self._reuse = True
        return value_estimation, model_variables
