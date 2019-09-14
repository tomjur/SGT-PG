import tensorflow as tf
import numpy as np

from autoregressive_model import AutoregressiveModel


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
        self.autoregressive_net = AutoregressiveModel(config)
        self.prediction_distribution, self.model_variables = self.autoregressive_net.create_network(
            self.start_inputs, self.goal_inputs, self.middle_inputs)

        # compute the loss
        log_likelihood = self.prediction_distribution.log_prob(self.middle_inputs)
        mean_log_likelihood = tf.reduce_mean(log_likelihood)
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

        merge_summaries = [
            tf.summary.scalar('predicted_cost', -self.prediction_loss),
            tf.summary.scalar('learn_rate', self.learn_rate_variable),
            tf.summary.scalar('log-likelihood', mean_log_likelihood)
        ]
        if self.initial_gradients_norm is not None:
            merge_summaries.append(
                tf.summary.scalar('initial_gradients_norm', self.initial_gradients_norm)
            )
        if self.clipped_gradients_norm is not None:
            merge_summaries.append(
                tf.summary.scalar('clipped_gradients_norm', self.clipped_gradients_norm)
            )

        # summaries for the critic optimization
        self.optimization_summaries = tf.summary.merge(merge_summaries)
        # this is the prediction over the entire subtree (element at index l contains the entire trajectory prediction
        # for a tree with l levels
        self.policy_tree_prediction = [
            self._get_policy_tree(self.start_inputs, self.goal_inputs, level, False) for level in range(0, self.levels)
        ]
        # and the test prediction which takes the mean not a sample
        self.test_policy_tree_prediction = [
            self._get_policy_tree(self.start_inputs, self.goal_inputs, level, True) for level in range(0, self.levels)
        ]

    def _reuse_policy_network(self, start_inputs, goal_inputs, take_mean):
        policy_distribution, model_variables = self.autoregressive_net.create_network(
            start_inputs, goal_inputs, take_mean=take_mean)
        assert len(model_variables) == 0
        return policy_distribution

    def _get_policy_tree(self, start_inputs, goal_inputs, level, take_mean):
        current_policy_distribution = self._reuse_policy_network(start_inputs, goal_inputs, take_mean)
        my_policy = [current_policy_distribution.sample()]
        if level == 0:
            return my_policy
        prefix_states = self._get_policy_tree(start_inputs, my_policy[0], level-1, take_mean)
        suffix_states = self._get_policy_tree(my_policy[0], goal_inputs, level-1, take_mean)
        return prefix_states + my_policy + suffix_states

    @staticmethod
    def optimize_by_loss(loss, parameters_to_optimize, learning_rate, gradient_limit):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss, parameters_to_optimize))
        if gradient_limit > 0.0:
            initial_gradients_norm = tf.global_norm(gradients)
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_limit, use_norm=initial_gradients_norm)
            clipped_gradients_norm = tf.global_norm(gradients)
        else:
            initial_gradients_norm, clipped_gradients_norm = None, None
        optimize_op = optimizer.apply_gradients(zip(gradients, variables))
        return initial_gradients_norm, clipped_gradients_norm, optimize_op

    def predict_policy(self, start_inputs, goal_inputs, level, sess, is_train):
        assert 1 <= level <= self.levels
        level = level-1
        ops = self.policy_tree_prediction[level] if is_train else self.test_policy_tree_prediction[level]
        return sess.run(ops, self._generate_feed_dictionary(start_inputs, goal_inputs))

    def train_policy(self, start_inputs, goal_inputs, middle_inputs, labels, sess, symmetric=True):
        feed_dictionary = self._generate_feed_dictionary(
            start_inputs, goal_inputs, middle_inputs=middle_inputs, labels=labels, symmetric=symmetric)
        ops = [
            self.optimization_summaries, self.prediction_loss, self.optimize
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
