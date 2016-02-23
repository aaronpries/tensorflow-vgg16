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


def evaluate(sess, images, labels, accuracy_op, validation_set, batch_size):
  batches = input_data.load_batches(validation_set, batch_size, finite=True)
  correct = 0.0
  total = 0
  for image_batch, label_batch in batches:
    accuracy = sess.run([accuracy_op], feed_dict={images: image_batch, labels: label_batch})
    correct += accuracy[0]
    total += image_batch.shape[0]
    print(accuracy[0])
    print(total)
  mean_correct = correct / float(total)
  return mean_correct


def main(saved, save_to, logdir, batch_size, steps):
  train_set, validation_set, test_set = input_data.make_split(DATA_FOLDER)
  train_batches = input_data.load_batches(train_set, batch_size)

  images, labels, pred_op, loss_op = fgo.init(fgo.FGO16(), batch_size)
  saver = tf.train.Saver()
  train_op, global_step = fgo.training(loss_op)
  summary_op = fgo.summaries(loss_op)
  accuracy_op = fgo.accuracy(pred_op, labels)

  with tf.Session() as sess:
    print("Starting FGO16")
    print("\tSaved model: %s" % saved)
    print("\tCheckpoints: %s" % save_to)
    print("\tSummaries: %s" % logdir)
    print("\tBatch size: %d" % batch_size)

    tf.initialize_all_variables().run()
    saver.restore(sess, saved)
    logdir = os.path.join(logdir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    summary_writer = tf.train.SummaryWriter(logdir, graph_def=sess.graph_def)

    for image_batch, label_batch in train_batches:
      feed_dict = {images: image_batch, labels: label_batch}
      train, loss, step, summary = sess.run([train_op, loss_op, global_step, summary_op], feed_dict=feed_dict)

      summary_writer.add_summary(summary, step)
      print("Step %d, loss: %f" % (step, loss))

      if step % 10 == 0:
        # saver.save(sess, save_to, global_step=step)
        accuracy = evaluate(sess, images, labels, accuracy_op, validation_set, batch_size)
        print("Accuracy on validation set: %.3f%%" % (accuracy*100))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('saved')
  parser.add_argument('export')
  parser.add_argument('--logdir', default="/tmp/fgo16")
  parser.add_argument('--batch', default=64, type=int)
  parser.add_argument('--steps', default=100000, type=int)
  args = parser.parse_args()
  main(args.saved, args.export, args.logdir, args.batch, args.steps)
