import tensorflow as tf
import os

from path_helper import init_dir
from log_utils import print_and_log


class ModelSaver:
    def __init__(self, saver_dir, max_to_keep, name, variables=None, print_log=True):
        self.saver_dir = saver_dir
        self.name = name
        self.print_log = print_log
        init_dir(saver_dir)

        if variables is None:
            variables = tf.global_variables()
        self.variables = variables

        self._saver = tf.compat.v1.train.Saver(
            var_list=self.variables, max_to_keep=max_to_keep, save_relative_paths=self.saver_dir)
        self._restore_path = None

        self.assertion_w_dictionary = {}

    def save(self, sess, global_step):
        self.assertion_w_dictionary = self.get_assertion_vars(sess)
        self._restore_path = self._saver.save(sess, os.path.join(self.saver_dir, self.name), global_step=global_step)
        if self.print_log:
            print_and_log(
                'saver {}: saved model from global step {} to {}'.format(self.name, global_step, self._restore_path))

    def restore(self, sess, restore_from=None):
        if restore_from is None:
            restore_from = self._restore_path
        if self.print_log:
            print_and_log('saver: {} restoring model from {}'.format(self.name, restore_from))
        self._saver.restore(sess, restore_from)
        # new_assertion_dictionary = self.get_assertion_vars(sess)
        # assert len(new_assertion_dictionary) == len(self.assertion_w_dictionary)
        # assert set(new_assertion_dictionary.keys()) == set(self.assertion_w_dictionary.keys())
        # for var_name in self.assertion_w_dictionary:
        #     assert np.all(np.equal(new_assertion_dictionary[var_name], self.assertion_w_dictionary[var_name]))

    def get_assertion_vars(self, sess):
        return {v.name: sess.run(v) for v in self.variables}
