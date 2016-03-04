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



def accuracy_once(sess, accuracy_op, loss_op, images_op, labels_op, batch):
  image_batch, label_batch = batch
  feed_dict = fgo.no_dropouts()
  feed_dict.update({images_op: image_batch, labels_op: label_batch})
  return sess.run([accuracy_op, loss_op], feed_dict=feed_dict)


def accuracy(sess, images_op, labels_op, accuracy_op, loss_op, batches, size):
  correct = 0.0
  total = 0.0
  pbar = tqdm(desc="Computing accuracy", total=size)
  for batch in batches:
    acc, loss = accuracy_once(sess, accuracy_op, loss_op, images_op, labels_op, batch)
    correct += acc
    total += batch[0].shape[0]
    pbar.update(batch[0].shape[0])
  return correct/total


def set_learning_rate(lr, accuracies):
  if len(accuracies) > 1 and accuracies[-1] < accuracies[-2]:
    return lr / 10.0
  return lr


def print_accuracy(accuracy):
  print("Accuracy on validation set: %.3f%%" % (100*accuracy))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data')
  parser.add_argument('saved')
  parser.add_argument('--save', default=None)
  parser.add_argument('--logdir', default=None)
  parser.add_argument('--batch', default=64, type=int)
  parser.add_argument('--steps', default=None, type=int)
  parser.add_argument('--save_steps', default=100, type=int)
  parser.add_argument('--eval_size', default=0, type=int)
  parser.add_argument('--gpu', default=1, type=bool)
  parser.add_argument('--eval_steps', default=50, type=int)
  param = parser.parse_args()

  train_set, validation_set, test_set = input_data.make_split(param.data)
  print("Train set (%d), validation set (%d), test set (%d)" % (len(train_set), len(validation_set), len(test_set)))
  train_batches = input_data.load_batches(train_set, param.batch, finite=False, shuffle=True, randflip=True, randshift=True, randcrop=True)

  if param.eval_size > 0:
    validation_set = validation_set[:param.eval_size]

  images_op, labels_op = fgo.inputs()
  logits_op = fgo.FGO16().graph(images_op, n_classes=61, wd=5e-4)
  prob_op = fgo.inference(logits_op)
  loss_op = fgo.cost(logits_op, labels_op, param.batch)

  saver = tf.train.Saver()

  train_op, global_step_op, lr_op = fgo.training(loss_op)
  accuracy_op = fgo.accuracy(prob_op, labels_op)
  summary_op = fgo.summaries(images_op, loss_op)

  accuracies = []

  def eval_accuracy(sess, file_set):
    batches = input_data.load_batches(file_set, param.batch, finite=True, shuffle=False, randflip=False, randshift=False, randcrop=False)
    return accuracy(sess, images_op, labels_op, accuracy_op, loss_op, batches, len(file_set))


  if not param.gpu:
    config = tf.ConfigProto(device_count={'GPU': 0})
  else:
    config = tf.ConfigProto()

  with tf.Session(config=config) as sess:
    print("Starting FGO16")

    tf.initialize_all_variables().run()
    saver.restore(sess, param.saved)

    # print_accuracy(eval_accuracy(sess, validation_set))

    if param.logdir:
      logdir = os.path.join(param.logdir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
      summary_writer = tf.train.SummaryWriter(logdir, graph_def=sess.graph_def, flush_secs=30)

    learning_rate = 1e-2

    for image_batch, label_batch in train_batches:
      feed_dict = {images_op: image_batch, labels_op: label_batch, lr_op: learning_rate}
      train, loss, step, summary = sess.run([train_op, loss_op, global_step_op, summary_op], feed_dict=feed_dict)
      epoch = math.floor(step*param.batch/len(train_set))
      print("Step %d (epoch %d, lr %f), loss: %f" % (step, epoch, learning_rate, loss))

      if param.logdir:
        summary_writer.add_summary(summary, step)

      if param.eval_size and step % param.eval_steps == 0:
        acc = eval_accuracy(sess, validation_set)
        accuracies.append(acc)
        learning_rate = set_learning_rate(learning_rate, accuracies)
        print_accuracy(acc)

      if param.save and step % param.save_steps == 0:
        path = saver.save(sess, param.save, global_step=step)
        print("Saved model to %s" % path)

      if param.steps and step >= param.steps:
        print("Reached (%d/%d) steps, stopping..." % (step, param.steps))
        break
