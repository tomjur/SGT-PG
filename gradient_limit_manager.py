import numpy as np
import tensorflow as tf


class GradientLimitManager:
    def __init__(self, gradient_limit, gradient_limit_quantile, gradient_history_limit):
        self.gradient_limit = gradient_limit
        self.gradient_limit_quantile = gradient_limit_quantile
        self.gradient_history_limit = gradient_history_limit

        self.use_gradient_limit = gradient_limit > 0.
        self.is_dynamic_limit = self.use_gradient_limit and gradient_limit_quantile is not None

        self._placeholders = {}
        self._gradient_history_by_key = {}

    def set_key(self, k):
        if self.use_gradient_limit:
            placeholder = tf.compat.v1.placeholder(tf.float32, (), name='{}_gradient_limit_input'.format(k))
        else:
            placeholder = None
        self._placeholders[k] = placeholder
        if self.is_dynamic_limit:
            self._gradient_history_by_key[k] = []
        return placeholder

    def update_gradient_limit(self, key, gradient_norm):
        if not self.is_dynamic_limit:
            return
        history = self._gradient_history_by_key[key]
        history.append(gradient_norm)
        if 0 < self.gradient_history_limit < len(history):
            history.pop(0)

    def update_feed_dict(self, feed_dict, key):
        if self.use_gradient_limit:
            feed_dict[self._placeholders[key]] = self._get_gradient_limit(key)

    def _get_gradient_limit(self, key):
        if not self.is_dynamic_limit:
            return self.gradient_limit
        history = self._gradient_history_by_key[key]
        if len(history) == 0:
            return self.gradient_limit
        return np.quantile(history, self.gradient_limit_quantile)
