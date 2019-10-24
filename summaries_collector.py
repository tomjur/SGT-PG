import os
import tensorflow as tf


class SummariesCollector:
    def __init__(self, summaries_dir, model_name):
        self._train_summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(summaries_dir, 'train_' + model_name))
        init_res = self._init_episode_summaries('train', self._train_summary_writer)
        self.write_train_success_summaries = init_res[0]
        self.write_train_optimization_summaries = init_res[1]

        self._test_summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(summaries_dir, 'test_' + model_name))
        init_res = self._init_episode_summaries('test', self._test_summary_writer)
        self.write_test_success_summaries = init_res[0]
        self.write_test_optimization_summaries = init_res[1]

    @staticmethod
    def _init_episode_summaries(prefix, summary_writer):
        success_rate_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        accumulated_cost_var = tf.Variable(0, trainable=False, dtype=tf.float32)

        summaries = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.scalar(prefix + '_success_rate', success_rate_var),
            tf.compat.v1.summary.scalar(prefix + '_accumulated_cost', accumulated_cost_var),
        ])

        def write_success_summaries(sess, global_step, success_rate, cost):
            summary_str = sess.run(summaries, feed_dict={
                success_rate_var: success_rate,
                accumulated_cost_var: cost,
            })

            summary_writer.add_summary(summary_str, global_step)
            summary_writer.flush()

        def write_optimization_summaries(summaries, global_step):
            summary_writer.add_summary(summaries, global_step)
            # for s in summaries:
            #     if s is not None:
            #         summary_writer.add_summary(s, global_step)
            summary_writer.flush()
            
        return write_success_summaries, write_optimization_summaries




