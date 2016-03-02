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


def accuracy_once(sess, accuracy_op, loss_op, images, labels, batch):
  image_batch, label_batch = batch
  feed_dict = fgo.no_dropouts()
  feed_dict.update({images: image_batch, labels: label_batch})
  return sess.run([accuracy_op, loss_op], feed_dict=feed_dict)


def accuracy(sess, images, labels, accuracy_op, loss_op, batches, size):
  correct = 0.0
  losses = 0.0
  total = 0.0
  count = 0.0
  pbar = tqdm(desc="Computing accuracy", total=size)
  for batch in batches:
    acc, loss = accuracy_once(sess, accuracy_op, loss_op, images, labels, batch)
    losses += loss
    correct += acc
    total += batch[0].shape[0]
    count += 1
    pbar.update(batch[0].shape[0])
  print("Accuracy on validation set: %.3f%% (%d/%d), loss: %f" % ((correct/total)*100, correct, total, losses/count))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('saved')
  parser.add_argument('--save', default=None)
  parser.add_argument('--logdir', default=None)
  parser.add_argument('--batch', default=64, type=int)
  parser.add_argument('--steps', default=1000, type=int)
  parser.add_argument('--eval_size', default=None, type=int)
  parser.add_argument('--gpu', default=1, type=bool)
  args = parser.parse_args()

  train_set, validation_set, test_set = input_data.make_split(DATA_FOLDER)
  train_batches = input_data.load_batches(train_set, args.batch, randflip=True, randshift=True, randcrop=True)

  known_set = input_data.keep_known(input_data.make_file_list(DATA_FOLDER))
  if args.eval_size:
    validation_set = validation_set[:args.eval_size]

  images, labels = fgo.inputs()
  logits_op = fgo.FGO16().graph(images, n_classes=61, wd=5e-4)
  prob_op = fgo.inference(logits_op)
  loss_op = fgo.cost(logits_op, labels, args.batch)

  saver = tf.train.Saver()

  train_op, global_step = fgo.training(loss_op, learning_rate=1e-2)
  accuracy_op = fgo.accuracy(prob_op, labels)
  summary_op = fgo.summaries(images, loss_op)

  def print_accuracy(sess, file_set):
    batches = input_data.load_batches(file_set, args.batch, finite=True, shuffle=True, randflip=False, randshift=False, randcrop=False)
    accuracy(sess, images, labels, accuracy_op, loss_op, batches, args.eval_size)

  if not args.gpu:
    config = tf.ConfigProto(device_count={'GPU': 0})
  else:
    config = tf.ConfigProto()

  with tf.Session(config=config) as sess:
    print("Starting FGO16")

    tf.initialize_all_variables().run()
    saver.restore(sess, args.saved)

    print_accuracy(sess, validation_set)

    if args.logdir:
      logdir = os.path.join(args.logdir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
      summary_writer = tf.train.SummaryWriter(logdir, graph_def=sess.graph_def, flush_secs=30)

    for image_batch, label_batch in train_batches:
      feed_dict = {images: image_batch, labels: label_batch}
      train, loss, step, summary = sess.run([train_op, loss_op, global_step, summary_op], feed_dict=feed_dict)
      print("Step %d, loss: %f" % (step, loss))

      if args.logdir:
        summary_writer.add_summary(summary, step)

      # acc, _ = accuracy_once(sess, accuracy_op, loss_op, images, labels, (image_batch, label_batch))
      # print(100*acc/args.batch)


      if step % math.ceil(args.steps/100) == 0:
        print_accuracy(sess, validation_set)

        if args.save:
          path = saver.save(sess, args.save, global_step=step)
          print("Saved model to %s" % path)

      if step >= args.steps:
        break
