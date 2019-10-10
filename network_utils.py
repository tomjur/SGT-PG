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
    initial_gradients_norm = tf.linalg.global_norm(gradients)
    if gradient_limit > 0.0:
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_limit, use_norm=initial_gradients_norm)
        clipped_gradients_norm = tf.linalg.global_norm(gradients)
    else:
        clipped_gradients_norm = initial_gradients_norm
    optimize_op = optimizer.apply_gradients(zip(gradients, variables))
    return initial_gradients_norm, clipped_gradients_norm, optimize_op
