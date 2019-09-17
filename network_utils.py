import tensorflow as tf


def get_activation(activation):
    if activation == 'relu':
        return tf.nn.relu
    if activation == 'tanh':
        return tf.nn.tanh
    if activation == 'elu':
        return tf.nn.elu
    return None


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
