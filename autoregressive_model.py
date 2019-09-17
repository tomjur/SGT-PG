import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from network_utils import get_activation


class AutoregressiveModel:
    def __init__(self, config, level):
        self.config = config
        self.state_size = self.config['model']['state_size']
        self.name_prefix = 'policy_level_{}'.format(level)
        self._reuse = False

    def create_network(self, start_inputs, goal_inputs, middle_inputs=None, take_mean=False):
        variable_count = len(tf.trainable_variables())
        activation = get_activation(self.config['policy']['activation'])
        network_layers = self.config['policy']['layers']
        base_std = self.config['policy']['base_std']
        learn_std = self.config['policy']['learn_std']

        if middle_inputs is None:
            split_middle_inputs = None
        else:
            split_middle_inputs = tf.split(middle_inputs, self.state_size, axis=1)

        # shift = tf.split((start_inputs + goal_inputs) * 0.5, self.state_size, axis=1)

        distributions, samples = [], []
        current_input = tf.concat((start_inputs, goal_inputs), axis=1)
        for s in range(self.state_size):
            current = current_input
            for i, layer_size in enumerate(network_layers):
                current = tf.layers.dense(
                    current, layer_size, activation=activation,
                    name='{}_autoregressive_{}_layer_{}'.format(self.name_prefix, s, i), reuse=self._reuse
                )

            if learn_std:
                normal_dist_parameters = tf.layers.dense(
                    current, 2, activation=None,
                    name='{}_autoregressive_{}_normal_dist_parameters'.format(self.name_prefix, s), reuse=self._reuse
                )
                split_normal_dist_parameters = tf.split(normal_dist_parameters, 2, axis=1)
                # bias = tf.squeeze(tf.tanh(split_normal_dist_parameters[0]), axis=1)
                bias = tf.squeeze(split_normal_dist_parameters[0], axis=1)
                bias = tf.maximum(tf.minimum(bias, 1.), -1.)
                # bias = tf.squeeze(shift[s] + split_normal_dist_parameters[0], axis=1)
                std = split_normal_dist_parameters[1]
                if base_std > 0.0:
                    std = tf.maximum(std, np.log(base_std))
                std = tf.squeeze(tf.exp(std), axis=1)
            else:
                normal_dist_parameters = tf.layers.dense(
                    current,  1, activation=None,
                    name='{}_autoregressive_{}_normal_dist_parameters'.format(self.name_prefix, s), reuse=self._reuse
                )
                # bias = tf.squeeze(shift[s] + normal_dist_parameters, axis=1)
                # bias = tf.squeeze(tf.tanh(normal_dist_parameters), axis=1)
                bias = tf.squeeze(normal_dist_parameters, axis=1)
                bias = tf.maximum(tf.minimum(bias, 1.), -1.)
                std = base_std

            # save the distribution
            current_prediction_distribution = tfp.distributions.TruncatedNormal(loc=bias, scale=std, low=-1., high=1.)
            distributions.append(current_prediction_distribution)

            # sample from the distribution
            if take_mean:
                # not really a sample, take the mean
                current_sample = current_prediction_distribution.mean()
            else:
                # sample
                current_sample = current_prediction_distribution.sample()
            current_sample = tf.expand_dims(current_sample, axis=-1)
            samples.append(current_sample)

            # generate the input for the next layer
            if split_middle_inputs is None:
                # the input is determined by the sample
                current_input = tf.concat((current_input, current_sample), axis=1)
            else:
                # the input is determined by the initial middle state
                current_input = tf.concat((current_input, split_middle_inputs[s]), axis=1)

        model_variables = tf.trainable_variables()[variable_count:]
        if self._reuse:
            assert len(model_variables) == 0
        else:
            self._reuse = True

        # wrap as a distribution
        auto_regressive_distribution = AutoregressiveDistribution(self.config, distributions, samples)
        return auto_regressive_distribution, model_variables


class AutoregressiveDistribution:
    def __init__(self, config, distributions, sample_ops):
        self.config = config
        self.distributions = distributions
        self.sample_ops = sample_ops

    def log_prob(self, x):
        # log_prob_ignore_pdf = self.config['policy']['log_prob_ignore_pdf']
        split_x = tf.split(x, len(self.distributions), axis=-1)
        log_probs = [
            # tf.expand_dims(
            #     tf.log(tf.maximum(self.distributions[d].prob(tf.squeeze(split_x[d], axis=1)), log_prob_ignore_pdf)),
            #     axis=1)
            # tf.expand_dims(
            #     tf.log(self.distributions[d].prob(tf.squeeze(split_x[d], axis=1)) + log_prob_ignore_pdf),
            #     axis=1)
            tf.expand_dims(self.distributions[d].log_prob(tf.squeeze(split_x[d], axis=1)), axis=1)
            for d in range(len(self.distributions))
        ]
        log_probs = tf.concat(log_probs, axis=1)
        # log_probs = tf.maximum(log_probs, np.log(log_prob_ignore_pdf))
        return tf.reduce_sum(log_probs, axis=1)

    def sample(self):
        return tf.concat(self.sample_ops, axis=1)
