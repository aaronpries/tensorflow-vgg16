import argparse
import os
import os.path
import skimage
from pprint import pprint
import tensorflow as tf
import utils
import sys
import random

import fgo


DATA_FOLDER = "imagenet"


def process_image(val):
  img = tf.image.decode_jpeg(val, channels=3)
  return tf.image.convert_image_dtype(img, tf.double)


def decode_image(filequeue):
  reader = tf.FixedLengthReader()
  fname, value = reader.read(filequeue)
  image = process_image(value)
  label = fname.split("/")[-2]
  return  image, label


def input_pipeline():
  filenames = tf.train.match_filenames_once(os.path.join(DATA_FOLDER, "n*/*"))
  filequeue = tf.train.string_input_producer(filenames, num_epochs=2)

  image, label = decode_image(filequeue)

  batch_size = 5
  min_after_dequeue = 10
  capacity = min_after_dequeue + 3 * batch_size

  image_batch, label_batch = tf.train.shuffle_batch([image, label],
    batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

  return image_batch, label_batch



def batches(files, batch_size, max_iter=1000):
  collection = utils.load_collection([f for f,l in files])
  labels = utils.load_labels([l for f,l in files])
  for i in range(max_iter):
    sample = random.sample(range(len(files)), batch_size)
    yield [collection[i] for i in sample], labels[sample, :]


def split(files):
  split_test = int(0.8*len(files))
  test_files = files[split_test:]
  rest = files[:split_test]
  split_validation = int(0.8*len(rest))
  train_files = rest[:split_validation]
  validation_files = rest[split_validation:]
  return train_files, validation_files, test_files


def input_pipeline_py(folder):
  files = [(os.path.join(folder, label, filename), label)
    for label in os.listdir(folder)
    for filename in os.listdir(os.path.join(folder, label))
  ]
  return split(files)


def main(saved, save_to, train_dir):
  batch_size = 256
  # var_names = [
  #   "fc6/weight:0", "fc6/bias:0",
  #   "fc7/weight:0", "fc7/bias:0",
  #   "fc8/weight:0", "fc8/bias:0"
  # ]
  # graph, tensors = fgo.load_graph("fgo16.tfmodel", var_names)

  # variables = [tf.Variable(tensors[i], name=var_names[i].split(":")[0], validate_shape=False) for i in range(len(tensors))]

  # [tf.Variable([], name=var_names[i].split(":")[0], validate_shape=False)
  #   for i in range(len(tensors))]
  # saver = tf.train.Saver(var_list=tensors)

  model = fgo.load_graph_empty()
  model.build_train(batch_size, dim=61)
  model.build_summary()

  train_set, validation_set, test_set = input_pipeline_py(DATA_FOLDER)
  train_batches = batches(train_set, batch_size)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())
    saver.restore(sess, saved)

    summary_writer = tf.train.SummaryWriter(train_dir, graph_def=sess.graph_def)

    i = 0
    for image_batch, label_batch in train_batches:
      print("iteration %d" % i)

      feed_dict = {model.images: image_batch, model.labels: label_batch}
      _, loss = sess.run([model.train, model.loss], feed_dict=feed_dict)
      i += 1

      print("loss: %f" % loss)

      path = saver.save(sess, save_to, global_step=i)
      print("saved into %s" % path)

      summary_str = sess.run(model.summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('saved')
  parser.add_argument('save_to')
  parser.add_argument('train_dir')
  args = parser.parse_args()
  main(args.saved, args.save_to, args.train_dir)
