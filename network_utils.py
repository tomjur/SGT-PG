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
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss, parameters_to_optimize))
    initial_gradients_norm = tf.linalg.global_norm(gradients)
    if gradient_limit is not None:
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_limit, use_norm=initial_gradients_norm)
        clipped_gradients_norm = tf.linalg.global_norm(gradients)
    else:
        clipped_gradients_norm = initial_gradients_norm
    grad_checks = [
        tf.check_numerics(g, 'gradients of {} are not numeric'.format(variables[i].name))
        for i, g in enumerate(gradients)
    ]
    with tf.control_dependencies(grad_checks):
        optimize_op = optimizer.apply_gradients(zip(gradients, variables))
        return initial_gradients_norm, clipped_gradients_norm, optimize_op
