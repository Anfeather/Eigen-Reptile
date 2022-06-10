"""
Training helpers for supervised meta-learning.
"""

import os
import time
import numpy as np
import tensorflow as tf

from .IER import IER
from .variables import weight_decay

# pylint: disable=R0913,R0914
def train(sess,
          model,
          train_set,
          test_set,
          save_dir,
          num_classes=5,
          ratio=0.0,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          eval_inner_batch_size=5,
          eval_inner_iters=50,
          eval_interval=10,
          weight_decay_rate=1,
          time_deadline=None,
          train_shots=None,
          transductive=False,
          IER_fn=IER,
          log_fn=print):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    saver = tf.train.Saver()

    IER = IER_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    accuracy_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('accuracy', accuracy_ph)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())

    x = []
    y = []
    count_1 = 0
    lp = 10.0
    for i in range(meta_iters):
        frac_done = i / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size


        IER.train_step_Eigen_O(train_set, model.input_ph, model.label_ph, model.minimize_op, ratio,
                        num_classes=num_classes,  num_shots=(train_shots or num_shots),
                        inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                        replacement=replacement,
                        meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)
    




        if i % eval_interval == 0:
            accuracies = []
            for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                correct = IER.evaluate(dataset, model.input_ph, model.label_ph,
                                           model.minimize_op, model.predictions,
                                           num_classes=num_classes, num_shots=num_shots,
                                           inner_batch_size=eval_inner_batch_size,
                                           inner_iters=eval_inner_iters, replacement=replacement)
                summary = sess.run(merged, feed_dict={accuracy_ph: correct/num_classes})
                writer.add_summary(summary, i)
                writer.flush()
                accuracies.append(correct / num_classes)
            log_fn('batch %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))
            if accuracies[1] > 0.4:
                count_1 = count_1 + 1

        if i % 1000 == 0:
            lp-=0.5
            x.append(i)
            y.append(count_1/(1000/eval_interval))
            print(count_1/(1000/eval_interval))
            count_1 = 0

        if i % 100 == 0 or i == meta_iters-1:
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
        if time_deadline is not None and time.time() > time_deadline:
            break
        # if i == 95000:
        #     break
