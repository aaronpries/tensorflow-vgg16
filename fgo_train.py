import argparse
import datetime
import os
import os.path
import skimage
from pprint import pprint
import tensorflow as tf
import utils
import sys
import random
import numpy as np

import fgo
import input_data


DATA_FOLDER = "imagenet"


def main(saved, save_to, logdir, batch_size, steps):
  print("Starting FGO16")
  print("\tSaved model: %s" % saved)
  print("\tCheckpoints: %s" % save_to)
  print("\tSummaries: %s" % logdir)
  print("\tBatch size: %d" % batch_size)

  train_set, validation_set, test_set = input_data.make_split(DATA_FOLDER)
  train_batches = input_data.batches(train_set, batch_size)

  images, labels, pred_op, loss_op = fgo.init(fgo.FGO16())
  saver = tf.train.Saver()
  train_op, global_step = fgo.training(loss_op)
  summary_op = fgo.summaries(loss_op)

  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver.restore(sess, saved)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    summary_writer = tf.train.SummaryWriter(os.path.join(logdir, now), graph_def=sess.graph_def)

    for image_batch, label_batch in train_batches:
      feed_dict = {images: image_batch, labels: label_batch}
      train, loss, step, summary = sess.run([train_op, loss_op, global_step, summary_op], feed_dict=feed_dict)

      summary_writer.add_summary(summary, step)
      print("Step %d, loss: %f" % (step, loss))

      # if step % 10 == 0:
        # saver.save(sess, save_to, global_step=step)
        # summary_str = sess.run(summary_op, feed_dict=feed_dict)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('saved')
  parser.add_argument('export')
  parser.add_argument('--logdir', default="/tmp/fgo16")
  parser.add_argument('--batch', default=64, type=int)
  parser.add_argument('--steps', default=100000, type=int)
  args = parser.parse_args()
  main(args.saved, args.export, args.logdir, args.batch, args.steps)
