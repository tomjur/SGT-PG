import os
import tensorflow as tf


class SummariesCollector:
    def __init__(self, summaries_dir, model_name):
        self._train_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train_' + model_name))
        init_res = self._init_episode_summaries('train', self._train_summary_writer)
        self.write_train_success_summaries, self.write_train_optimization_summaries = init_res

        self._test_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'test_' + model_name))
        init_res = self._init_episode_summaries('test', self._test_summary_writer)
        self.write_test_success_summaries, self.write_test_optimization_summaries = init_res

    @staticmethod
    def _init_episode_summaries(prefix, summary_writer):
        success_rate_var = tf.Variable(0, trainable=False, dtype=tf.float32)

        summaries = tf.summary.merge([
            tf.summary.scalar(prefix + '_success_rate', success_rate_var),
        ])

        def write_success_summaries(sess, global_step, success_rate):
            summary_str = sess.run(summaries, feed_dict={
                success_rate_var: success_rate,
            })

            summary_writer.add_summary(summary_str, global_step)
            summary_writer.flush()

        # episodes_played_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        # successful_episodes_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        # mean_episode_length_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        #
        # summaries = tf.summary.merge([
        #     tf.summary.scalar(prefix + '_episodes_played', episodes_played_var),
        #     tf.summary.scalar(prefix + '_successful_episodes', successful_episodes_var / episodes_played_var),
        #     tf.summary.scalar(prefix + '_mean_episode_length', mean_episode_length_var),
        # ])
        #
        # def write_episode_summaries(sess, global_step, episodes_played, successful_episodes, mean_episode_length):
        #     summary_str = sess.run(summaries, feed_dict={
        #         episodes_played_var: episodes_played,
        #         successful_episodes_var: successful_episodes,
        #         mean_episode_length_var: mean_episode_length
        #     })
        #
        #     summary_writer.add_summary(summary_str, global_step)
        #     summary_writer.flush()

        def write_optimization_summaries(summaries, global_step):
            summary_writer.add_summary(summaries, global_step)
            # for s in summaries:
            #     if s is not None:
            #         summary_writer.add_summary(s, global_step)
            summary_writer.flush()
            
        return write_success_summaries, write_optimization_summaries




