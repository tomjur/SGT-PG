import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from gradient_limit_manager import GradientLimitManager
from network_utils import optimize_by_loss, get_activation


class Network:
    def __init__(self, config, game):
        self.config = config
        self.state_size = game.get_state_size()
        self.levels = self.config['model']['levels']

        self.start_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), name='start_inputs')
        self.goal_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), name='goal_inputs')
        self.middle_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.state_size), name='middle_inputs')
        self.label_inputs = tf.compat.v1.placeholder(tf.float32, (None, 1), name='label_inputs')

        self.gradient_limit_manager = GradientLimitManager(
            gradient_limit=config['policy']['gradient_limit'],
            gradient_limit_quantile=config['policy']['gradient_limit_quantile'],
            gradient_history_limit=config['policy']['gradient_history_limit'],
        )

        # network is comprised of several policies
        self.policy_networks = {}
        previous_policy = None
        for level in range(1, 1+self.levels):
            # create policies
            current_policy = PolicyNetwork(
                config, level, self.state_size, self.start_inputs, self.goal_inputs, self.middle_inputs,
                self.label_inputs, self.gradient_limit_manager, previous_policy
            )
            self.policy_networks[level] = current_policy
            previous_policy = current_policy

        # this is the prediction over the entire subtree (element at index l contains the entire trajectory prediction
        # for a tree with l levels
        self.policy_tree_prediction = {
            level: self._get_policy_tree(self.start_inputs, self.goal_inputs, level, False)
            for level in range(1, 1 + self.levels)
        }
        # and the test prediction which takes the mean not a sample
        self.test_policy_tree_prediction = {
            level: self._get_policy_tree(self.start_inputs, self.goal_inputs, level, True)
            for level in range(1, 1 + self.levels)
        }

    def _get_policy_tree(self, start_inputs, goal_inputs, level, take_mean):
        current_policy_distribution = self.policy_networks[level].reuse_policy_network(
            start_inputs, goal_inputs)
        if take_mean:
            current_state = [current_policy_distribution.mean()]
        else:
            current_state = [current_policy_distribution.sample()]
        if level == 1:
            return current_state
        # when training a policy of level L, the prediction of levels 1... L-1 are not stochastic
        prefix_states = self._get_policy_tree(start_inputs, current_state[0], level-1, True)
        suffix_states = self._get_policy_tree(current_state[0], goal_inputs, level-1, True)
        return prefix_states + current_state + suffix_states

    def predict_policy(self, start_inputs, goal_inputs, level, sess, is_train):
        assert 1 <= level <= self.levels
        tree = self.policy_tree_prediction if is_train else self.test_policy_tree_prediction
        return sess.run(tree[level], self._generate_feed_dictionary(start_inputs, goal_inputs))

    def train_policy(self, level, start_inputs, goal_inputs, middle_inputs, labels, sess, symmetric=True):
        assert 1 <= level <= self.levels
        network = self.policy_networks[level]
        feed_dictionary = self._generate_feed_dictionary(
            start_inputs, goal_inputs, middle_inputs=middle_inputs, labels=labels, symmetric=symmetric)
        self.gradient_limit_manager.update_feed_dict(feed_dictionary, network.name_prefix)
        ops = [
            network.initial_gradients_norm, network.clipped_gradients_norm, network.optimization_summaries,
            network.cost_loss, network.optimize
        ]
        result = sess.run(ops, feed_dictionary)
        initial_gradients = result[0]
        self.gradient_limit_manager.update_gradient_limit(network.name_prefix, initial_gradients)
        return result

    def decrease_learn_rates(self, sess, level):
        assert 1 <= level <= self.levels
        return sess.run([self.policy_networks[level].decrease_learn_rate_op])

    def decrease_base_std(self, sess, level):
        assert 1 <= level <= self.levels
        return sess.run(self.policy_networks[level].decrease_base_std_op)

    def get_learn_rates(self, sess, level_limit=None):
        if level_limit is None:
            level_limit = self.levels
        assert 1 <= level_limit <= self.levels
        return sess.run([self.policy_networks[level].learn_rate_variable for level in range(1, 1+level_limit)])

    def get_base_stds(self, sess, level_limit=None):
        if level_limit is None:
            level_limit = self.levels
        assert 1 <= level_limit <= self.levels
        ops = [
            self.policy_networks[level].base_std_variable for level in range(1, 1 + level_limit)
        ]
        return sess.run(ops)

    def init_policy_from_lower_level(self, sess, current_level):
        assert 1 < current_level <= self.levels
        sess.run(self.policy_networks[current_level].assign_from_last_policy_ops)

    def update_baseline_policy(self, sess, current_level):
        assert 1 <= current_level <= self.levels
        sess.run(self.policy_networks[current_level].assign_to_baseline_ops)

    def get_all_variables(self):
        model_variables = []
        for level in self.policy_networks:
            model_variables.extend(self.policy_networks[level].model_variables)
        return model_variables

    def _generate_feed_dictionary(self, start_inputs, goal_inputs, middle_inputs=None, labels=None, symmetric=False):
        start_inputs_ = np.array(start_inputs)
        goal_inputs_ = np.array(goal_inputs)

        if middle_inputs is None:
            middle_inputs_ = None
        else:
            middle_inputs_ = np.array(middle_inputs)

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
            if middle_inputs_ is not None:
                middle_inputs_ = np.concatenate((middle_inputs_, middle_inputs_), axis=0)
            if labels_ is not None:
                labels_ = np.concatenate((labels_, labels_), axis=0)

        feed_dictionary = {
            self.start_inputs: start_inputs_,
            self.goal_inputs: goal_inputs_,
        }

        if middle_inputs_ is not None:
            feed_dictionary[self.middle_inputs] = middle_inputs_

        if labels_ is not None:
            feed_dictionary[self.label_inputs] = labels_

        return feed_dictionary


class PolicyNetwork:
    def __init__(self, config, level, state_size, start_inputs, goal_inputs, middle_inputs, label_inputs,
                 gradient_limit_manager, previous_policy):
        self.config = config
        self.name_prefix = 'policy_level_{}'.format(level)
        self.state_size = state_size
        self._reuse = False

        self.start_inputs = start_inputs
        self.goal_inputs = goal_inputs
        self.middle_inputs = middle_inputs
        self.label_inputs = label_inputs

        self.gradient_limit_placeholder = gradient_limit_manager.set_key(self.name_prefix)

        self.base_std_variable = tf.Variable(
            self.config['policy']['base_std'], trainable=False, name='base_std_variable')
        new_base_std = self.config['policy']['std_decrease_rate'] * self.base_std_variable
        self.decrease_base_std_op = tf.compat.v1.assign(self.base_std_variable, new_base_std)

        # get the prediction distribution
        self.prediction_distribution, self.model_variables = self._create_network(self.start_inputs, self.goal_inputs)

        # to update from last level:
        if previous_policy is not None:
            self.assign_from_last_policy_ops = self.get_assignment_between_policies(
                previous_policy.model_variables, self.model_variables
            )
        else:
            self.assign_from_last_policy_ops = None

        # get the baseline prediction distribution (remains fixed during optimization)
        self.baseline_prediction_distribution, self.baseline_model_variables = self._create_network(
            self.start_inputs, self.goal_inputs, is_baseline=True)

        # to update baseline to the optimized:
        self.assign_to_baseline_ops = self.get_assignment_between_policies(
            self.model_variables, self.baseline_model_variables
        )

        # compute the policies ratio
        log_likelihood = self.prediction_distribution.log_prob(self.middle_inputs)
        log_likelihood_baseline = self.baseline_prediction_distribution.log_prob(self.middle_inputs)
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
