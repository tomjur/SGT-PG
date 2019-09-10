import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Network:
    def __init__(self, config):
        self.config = config
        self.state_size = self.config['model']['state_size']
        self.levels = self.config['model']['levels']

        self.start_inputs = tf.placeholder(tf.float32, (None, self.state_size), name='start_inputs')
        self.goal_inputs = tf.placeholder(tf.float32, (None, self.state_size), name='goal_inputs')
        self.middle_inputs = tf.placeholder(tf.float32, (None, self.state_size), name='middle_inputs')
        self.label_inputs = tf.placeholder(tf.float32, (None, 1), name='label_inputs')

        # get the prediction distribution
        self.prediction_distribution, self.model_variables = self._create_policy_network(
            False, self.start_inputs, self.goal_inputs)

        # compute the loss
        log_likelihood = self.prediction_distribution.log_prob(self.middle_inputs)
        mean_log_likelihood = tf.reduce_mean(log_likelihood)
        log_likelihood = tf.maximum(log_likelihood, np.log(0.01))
        self.prediction_loss = tf.reduce_mean(tf.expand_dims(log_likelihood, axis=-1) * self.label_inputs)

        # optimize
        self.learn_rate_variable = tf.Variable(
            self.config['policy']['learning_rate'], trainable=False, name='learn_rate_variable')
        new_learn_rate = tf.maximum(self.config['policy']['learning_rate_minimum'],
                                    self.config['policy']['learning_rate_decrease_rate'] * self.learn_rate_variable)
        self.decrease_learn_rate_op = tf.assign(self.learn_rate_variable, new_learn_rate)

        self.initial_gradients_norm, self.clipped_gradients_norm, self.optimize = \
            Network.optimize_by_loss(
                self.prediction_loss, self.model_variables, self.learn_rate_variable,
                self.config['policy']['gradient_limit']
            )

        # summaries for the critic optimization
        self.optimization_summaries = tf.summary.merge([
            tf.summary.scalar('predicted_cost', self.prediction_loss),
            tf.summary.scalar('initial_gradients_norm', self.initial_gradients_norm),
            tf.summary.scalar('clipped_gradients_norm', self.clipped_gradients_norm),
            tf.summary.scalar('learn_rate', self.learn_rate_variable),
            tf.summary.scalar('log-likelihood', mean_log_likelihood)
        ])
        # this is the prediction over the entire subtree (element at index l contains the entire trajectory prediction
        # for a tree with l levels
        self.policy_tree_prediction = [
            self._get_policy_tree(self.start_inputs, self.goal_inputs, level) for level in range(0, self.levels)
        ]

    def _reuse_policy_network(self, start_inputs, goal_inputs):
        policy_distribution, model_variables = self._create_policy_network(True, start_inputs, goal_inputs)
        assert len(model_variables) == 0
        return policy_distribution

    def _get_policy_tree(self, start_inputs, goal_inputs, level):
        current_policy_distribution = self._reuse_policy_network(start_inputs, goal_inputs)
        my_policy = [current_policy_distribution.sample()]
        if level == 0:
            return my_policy
        prefix_states = self._get_policy_tree(start_inputs, my_policy[0], level-1)
        suffix_states = self._get_policy_tree(my_policy[0], goal_inputs, level-1)
        return prefix_states + my_policy + suffix_states

    def _create_policy_network(self, reuse_flag, start_inputs, goal_inputs):
        variable_count = len(tf.trainable_variables())
        activation = Network.get_activation(self.config['policy']['activation'])
        network_layers = self.config['policy']['layers']
        current = tf.concat((start_inputs, goal_inputs), axis=1)
        for i, layer_size in enumerate(network_layers):
            current = tf.layers.dense(
                current, layer_size, activation=activation, name='policy_{}'.format(i), reuse=reuse_flag
            )

        # predict a mixture of gaussians:
        number_of_gaussians = self.config['policy']['number_of_gaussians']
        mixing_logits = tf.layers.dense(
            current, number_of_gaussians, activation=None, name='policy_mixing', reuse=reuse_flag
        )
        mixture_selection = tfp.distributions.Categorical(logits=mixing_logits)

        shift = (start_inputs + goal_inputs) * 0.5

        normal_dist_parameters = tf.layers.dense(
            current, number_of_gaussians * self.state_size * 2, activation=None, name='policy_normal_dist_parameters',
            reuse=reuse_flag
        )
        split_normal_dist_parameters = tf.split(normal_dist_parameters, number_of_gaussians * 2, axis=1)
        normal_distributions = [
            tfp.distributions.MultivariateNormalDiagWithSoftplusScale(
                loc=shift + split_normal_dist_parameters[2 * i], scale_diag=split_normal_dist_parameters[2 * i + 1]
            ) for i in range(number_of_gaussians)
        ]

        # normal_dist_parameters = tf.layers.dense(
        #     current, number_of_gaussians * self.state_size, activation=None, name='policy_normal_dist_parameters',
        #     reuse=reuse_flag
        # )
        # split_normal_dist_parameters = tf.split(normal_dist_parameters, number_of_gaussians, axis=1)
        # normal_distributions = [
        #     # tfp.distributions.MultivariateNormalDiag(loc=split_normal_dist_parameters[i],scale_identity_multiplier=0.01)
        #     tfp.distributions.MultivariateNormalDiag(loc=shift + split_normal_dist_parameters[i], scale_identity_multiplier=0.01)
        #     for i in range(number_of_gaussians)
        # ]

        mixture_distribution = tfp.distributions.Mixture(
            cat=mixture_selection, components=normal_distributions
        )

        model_variables = tf.trainable_variables()[variable_count:]
        return mixture_distribution, model_variables

    @staticmethod
    def get_activation(activation):
        if activation == 'relu':
            return tf.nn.relu
        if activation == 'tanh':
            return tf.nn.tanh
        if activation == 'elu':
            return tf.nn.elu
        return None

    @staticmethod
    def optimize_by_loss(loss, parameters_to_optimize, learning_rate, gradient_limit):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss, parameters_to_optimize))
        initial_gradients_norm = tf.global_norm(gradients)
        if gradient_limit > 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_limit, use_norm=initial_gradients_norm)
        clipped_gradients_norm = tf.global_norm(gradients)
        optimize_op = optimizer.apply_gradients(zip(gradients, variables))
        return initial_gradients_norm, clipped_gradients_norm, optimize_op

    def predict_policy(self, start_inputs, goal_inputs, level, sess):
        assert 1 <= level <= self.levels
        level = level-1
        ops = self.policy_tree_prediction[level]
        return sess.run(ops, self._generate_feed_dictionary(start_inputs, goal_inputs))

    def train_policy(self, start_inputs, goal_inputs, middle_inputs, labels, sess, symmetric=True):
        feed_dictionary = self._generate_feed_dictionary(
            start_inputs, goal_inputs, middle_inputs=middle_inputs, labels=labels, symmetric=symmetric)
        ops = [
            self.optimization_summaries, self.prediction_loss, self.initial_gradients_norm, self.clipped_gradients_norm,
            self.optimize
        ]
        return sess.run(ops, feed_dictionary)

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
