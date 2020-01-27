import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from gradient_limit_manager import GradientLimitManager
from network_utils import optimize_by_loss, get_activation


class NetworkSequential:
    def __init__(self, config, state_size, action_size, is_rollout_agent=False):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size

        self.current_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), name='current_inputs')
        self.goal_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), name='goal_inputs')
        self.action_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.action_size), name='action_inputs')
        self.label_inputs = tf.compat.v1.placeholder(tf.float32, (None, 1), name='label_inputs')

        # network is comprised of policy and value nets
        self.policy_network = PolicyNetwork(
            config, self.state_size, self.action_size, self.current_inputs, self.goal_inputs, self.action_inputs,
            self.label_inputs, is_rollout_agent
        )
        # policy predictions
        policy_distribution = self.policy_network.prediction_distribution
        self.train_prediction = policy_distribution.sample()
        self.test_prediction = policy_distribution.mean()

        # initialize value networks if required
        if is_rollout_agent:
            self.value_network = None
        else:
            self.value_network = ValueNetwork(
                config, self.state_size, self.current_inputs, self.goal_inputs, self.label_inputs,
            )

    def predict_policy(self, current_inputs, goal_inputs, sess, is_train):
        op = self.train_prediction if is_train else self.train_prediction
        return sess.run(op, self._generate_feed_dictionary(current_inputs, goal_inputs))

    def predict_value(self, current_inputs, goal_inputs, sess):
        return sess.run(
            self.value_network.value_estimation, self._generate_feed_dictionary(current_inputs, goal_inputs)
        )

    def train_policy(self, current_inputs, goal_inputs, action_inputs, labels, sess):
        network = self.policy_network
        feed_dictionary = self._generate_feed_dictionary(
            current_inputs, goal_inputs, action_inputs=action_inputs, labels=labels)
        network.gradient_limit_manager.update_feed_dict(feed_dictionary, network.name_prefix)
        ops = [
            network.initial_gradients_norm, network.clipped_gradients_norm, network.optimization_summaries,
            network.cost_loss, network.optimize
        ]
        result = sess.run(ops, feed_dictionary)
        initial_gradients = result[0]
        network.gradient_limit_manager.update_gradient_limit(network.name_prefix, initial_gradients)
        return result[2:]

    def train_value_estimation(self, current_inputs, goal_inputs, labels, sess):
        network = self.value_network
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
        return sess.run([self.policy_network.decrease_learn_rate_op, self.value_network.decrease_learn_rate_op])

    def decrease_base_std(self, sess):
        return sess.run(self.policy_network.decrease_base_std_op)

    def get_learn_rate(self, sess):
        return sess.run([self.policy_network.learn_rate_variable, self.value_network.learn_rate_variable])

    def get_base_std(self, sess):
        return sess.run(self.policy_network.base_std_variable)

    def update_baseline_policy(self, sess):
        sess.run(self.policy_network.assign_to_baseline_ops)

    def get_all_variables(self):
        return self.policy_network.model_variables + self.value_network.model_variables

    def get_policy_weights(self, sess):
        return sess.run([self.policy_network.model_variables, self.policy_network.base_std_variable])

    def set_policy_weights(self, sess, variables):
        weights, std = variables
        feed = {
            self.policy_network.weights_placeholders[var.name]: weights[i]
            for i, var in enumerate(self.policy_network.model_variables)
        }
        feed[self.policy_network.std_placeholder] = std
        sess.run([self.policy_network.weights_assign_ops, self.policy_network.std_assign_op], feed)

    def _generate_feed_dictionary(self, current_inputs, goal_inputs, action_inputs=None, labels=None):
        current_inputs_ = np.array(current_inputs)
        goal_inputs_ = np.array(goal_inputs)

        if action_inputs is None:
            action_inputs_ = None
        else:
            action_inputs_ = np.array(action_inputs)

        if labels is None:
            labels_ = None
        else:
            labels_ = np.array(labels)
            if len(labels_.shape) == 1:
                labels_ = np.expand_dims(labels_, axis=1)

        feed_dictionary = {
            self.current_inputs: current_inputs_,
            self.goal_inputs: goal_inputs_,
        }

        if action_inputs_ is not None:
            feed_dictionary[self.action_inputs] = action_inputs_

        if labels_ is not None:
            feed_dictionary[self.label_inputs] = labels_

        return feed_dictionary


class PolicyNetwork:
    def __init__(self, config, state_size, action_size, current_state_inputs, goal_inputs, action_inputs, label_inputs,
                 is_rollout_agent):
        self.config = config
        self.name_prefix = 'sequential_policy'
        self.state_size = state_size
        self.action_size = action_size
        self._reuse = False

        self.current_state_inputs = current_state_inputs
        self.goal_inputs = goal_inputs
        self.action_inputs = action_inputs
        self.label_inputs = label_inputs

        self.gradient_limit_manager = GradientLimitManager(
            gradient_limit=config['policy']['gradient_limit'],
            gradient_limit_quantile=config['policy']['gradient_limit_quantile'],
            gradient_history_limit=config['policy']['gradient_history_limit'],
        )
        self.gradient_limit_placeholder = self.gradient_limit_manager.set_key(self.name_prefix)

        self.base_std_variable = tf.Variable(
            self.config['policy']['base_std'], trainable=False, name='base_std_variable')
        new_base_std = self.config['policy']['std_decrease_rate'] * self.base_std_variable
        self.decrease_base_std_op = tf.compat.v1.assign(self.base_std_variable, new_base_std)

        # get the prediction distribution
        self.prediction_distribution, self.model_variables = self._create_network(
            self.current_state_inputs, self.goal_inputs)

        # create placeholders and assign ops to set these weights manually (used by rollout agents)
        self.weights_placeholders = {
            var.name: tf.placeholder(tf.float32, var.get_shape()) for var in self.model_variables
        }
        self.weights_assign_ops = [
            tf.assign(var, self.weights_placeholders[var.name])
            for var in self.model_variables
        ]
        self.std_placeholder = tf.placeholder(tf.float32,)
        self.std_assign_op = tf.assign(self.base_std_variable, self.std_placeholder)

        if is_rollout_agent:
            return

        # get the baseline prediction distribution (remains fixed during optimization)
        self.baseline_prediction_distribution, self.baseline_model_variables = self._create_network(
            self.current_state_inputs, self.goal_inputs, is_baseline=True)

        # to update baseline to the optimized:
        self.assign_to_baseline_ops = self.get_assignment_between_policies(
            self.model_variables, self.baseline_model_variables
        )

        # compute the policies ratio
        log_likelihood = self.prediction_distribution.log_prob(self.action_inputs)
        log_likelihood_baseline = self.baseline_prediction_distribution.log_prob(self.action_inputs)
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

    def reuse_policy_network(self, current_state_inputs, goal_inputs):
        policy_distribution, model_variables = self._create_network(current_state_inputs, goal_inputs)
        assert len(model_variables) == 0
        return policy_distribution

    def _create_network(self, current_state_inputs, goal_inputs, is_baseline=False):
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

        base_std = self.base_std_variable
        current_input = tf.concat((current_state_inputs, goal_inputs), axis=1)

        current = current_input
        for i, layer_size in enumerate(network_layers):
            current = tf.layers.dense(
                current, layer_size, activation=activation, name='{}_layer_{}'.format(name_prefix, i), reuse=reuse,
            )

        if learn_std:
            normal_dist_parameters = tf.layers.dense(
                current, self.action_size * 2, activation=None,
                name='{}_normal_dist_parameters'.format(name_prefix), reuse=reuse,
            )
            split_normal_dist_parameters = tf.split(normal_dist_parameters, 2, axis=1)
            bias = split_normal_dist_parameters[0]
            std = split_normal_dist_parameters[1]
            std = tf.math.softplus(std)
            std = std + base_std
        else:
            normal_dist_parameters = tf.layers.dense(
                current,  self.action_size, activation=None,
                name='{}_normal_dist_parameters'.format(name_prefix), reuse=reuse,
            )
            bias = normal_dist_parameters
            std = [base_std] * self.action_size

        if self.config['policy']['bias_activation_is_tanh']:
            bias = tf.tanh(bias)

        bias_towards_goal = self.config['policy']['bias_towards_goal']
        if bias_towards_goal is not None:
            bias = bias + current_state_inputs * (1. - bias_towards_goal) + goal_inputs * bias_towards_goal

        distribution = tfp.distributions.MultivariateNormalDiag(loc=bias, scale_diag=std)
        model_variables = tf.compat.v1.trainable_variables()[variable_count:]
        if reuse:
            assert len(model_variables) == 0
        elif not is_baseline:
            self._reuse = True
        return distribution, model_variables


class ValueNetwork:
    def __init__(self, config, state_size, current_state_inputs, goal_inputs, label_inputs):
        self.config = config
        self.name_prefix = 'sequential_value_estimator'
        self.state_size = state_size
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
