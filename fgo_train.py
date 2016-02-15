import argparse
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
  def maybe(i):
    try: return (collection[i], labels[i])
    except IOError: return None

  for i in range(max_iter):
    sample = random.sample(range(len(files)), batch_size)
    l = [maybe(i) for i in sample]
    s = list(filter(lambda x: x is not None and x[0].shape == (224,224,3) and x[1].shape == (61,), l))
    im, lab = zip(*s)
    im = np.stack(im, axis=0)
    lab = np.stack(lab, axis=0)
    yield im, lab
    

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


def main(saved, save_to, train_dir, batch_size):
  print("Starting FGO16")
  print("\tSaved model: %s" % saved)
  print("\tCheckpoints: %s" % save_to)
  print("\tSummaries: %s" % train_dir)
  print("\tBatch size: %d" % batch_size)

  model = fgo.load_graph_empty()
  model.build_train(dim=61)
  model.build_summary()

  variables = [v for v in tf.all_variables() if not "Momentum" in v.name]
  saver = tf.train.Saver(var_list=variables)

  train_set, validation_set, test_set = input_pipeline_py(DATA_FOLDER)
  train_batches = batches(train_set, batch_size, int(5e5))

  def save(sess, step):
    saver.save(sess, save_to, global_step=step)

  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver.restore(sess, saved)
    summary_writer = tf.train.SummaryWriter(train_dir, graph_def=sess.graph_def)

    i = 1
    # try:
    for image_batch, label_batch in train_batches:
      feed_dict = {model.images: image_batch, model.labels: label_batch}
      _, loss = sess.run([model.train, model.loss], feed_dict=feed_dict)
      print("iteration %d, loss: %f" % (i, loss))

      if i % 50 == 0:
        save(sess, i)
        summary_str = sess.run(model.summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, i)

      i += 1
    # except Exception as e:
      # save(sess, i)
      # raise e


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('saved')
  parser.add_argument('save_to')
  parser.add_argument('train_dir')
  parser.add_argument('--batch', default=256, type=int)
  args = parser.parse_args()
  main(args.saved, args.save_to, args.train_dir, args.batch)

