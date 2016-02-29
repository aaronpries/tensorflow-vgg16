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
import math
import numpy as np
from tqdm import tqdm

import fgo
import input_data


DATA_FOLDER = "imagenet"


def accuracy_once(sess, accuracy_op, images, labels, batch, feed_dict={}):
  image_batch, label_batch = batch
  feed_dict.update({images: image_batch, labels: label_batch})
  return sess.run([accuracy_op], feed_dict=feed_dict)[0]


def accuracy(sess, images, labels, accuracy_op, file_set, batch_size):
  batches = input_data.load_batches(file_set, batch_size, finite=True, shuffle=True, randflip=False, randshift=False, randcrop=False)
  correct = 0.0
  total = 0
  pbar = tqdm(desc="Computing accuracy", total=len(file_set))
  feed_dict = fgo.no_dropouts()
  for batch in batches:
    acc = accuracy_once(sess, accuracy_op, images, labels, batch, feed_dict=feed_dict)
    correct += acc
    total += batch[0].shape[0]
    pbar.update(batch_size)
  mean_correct = correct / float(total)
  return mean_correct, correct, total


def main(saved, save_to, logdir, batch_size, steps, eval_size):
  train_set, validation_set, test_set = input_data.make_split(DATA_FOLDER)
  train_batches = input_data.load_batches(train_set, batch_size, randflip=True, randshift=True, randcrop=True)

  known_set = input_data.keep_known(input_data.make_file_list(DATA_FOLDER))
  if eval_size:
    validation_set = random.sample(validation_set, eval_size)
    known_set = random.sample(known_set, eval_size)

  images, labels, pred_op, loss_op = fgo.init(fgo.FGO16(), batch_size, n_classes=61)
  saver = tf.train.Saver()
  train_op, global_step = fgo.training(loss_op)
  summary_op = fgo.summaries(loss_op)
  accuracy_op = fgo.accuracy(pred_op, labels)

  def print_accuracy(sess, file_set):
    acc, correct, total = accuracy(sess, images, labels, accuracy_op, file_set, batch_size)
    print("Accuracy on validation set: %.3f%% (%d/%d)" % (acc*100, correct, total))

  with tf.Session() as sess:
    print("Starting FGO16")

    tf.initialize_all_variables().run()
    saver.restore(sess, saved)

    print_accuracy(sess, validation_set)

    # acc, correct, total = accuracy(sess, images, labels, accuracy_op, known_set, batch_size, max_eval_size=eval_size)
    # print("Accuracy on validation set: %.3f%% (%d/%d)" % (acc*100, correct, total))
    # return

    logdir = os.path.join(logdir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    summary_writer = tf.train.SummaryWriter(logdir, graph_def=sess.graph_def, flush_secs=30)

    for image_batch, label_batch in train_batches:
      feed_dict = {images: image_batch, labels: label_batch}
      train, loss, step, summary = sess.run([train_op, loss_op, global_step, summary_op], feed_dict=feed_dict)

      summary_writer.add_summary(summary, step)
      print("Step %d, loss: %f" % (step, loss))

      acc = accuracy_once(sess, accuracy_op, images, labels, (image_batch, label_batch))
      print(acc/batch_size)

      if step % math.ceil(steps/100) == 0:
        print_accuracy(sess, validation_set)
        # path = saver.save(sess, save_to, global_step=step)
        # print("Saved model to %s" % path)

      if step >= steps:
        break


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('saved')
  parser.add_argument('export')
  parser.add_argument('--logdir', default="/tmp/fgo16")
  parser.add_argument('--batch', default=64, type=int)
  parser.add_argument('--steps', default=1000, type=int)
  parser.add_argument('--eval_size', default=None, type=int)
  args = parser.parse_args()
  main(args.saved, args.export, args.logdir, args.batch, args.steps, args.eval_size)
