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
import skflow

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

def load_data(files):
  random.shuffle(files)
  image_files, label_files = zip(*files)
  images = utils.load_collection(image_files)
  labels = utils.load_labels_indices(label_files)

  class Rand:
    def __init__(self, N, seed=None, calls=2):
      self.N = N
      self.calls = calls
      random.seed(seed)
      self.change()

    def change(self):
      self.x = random.randint(0, self.N)
      self.count = 0

    def get(self):
      self.count += 1
      x = self.x
      if self.count >= self.calls:
        self.change()
      return x

  def is_valid(i):
    try:
      im = images[i]
      return im.shape == (224,224,3)
    except IOError: return False

  rand = Rand(len(files))
    
  def load_X():
    while True:
      i = rand.get()
      if is_valid(i):
        im = images[i]
        if len(im.shape) == 2:
          im = np.expand_dims(im, 2)
        yield im

  def load_y():
    while True:
      i = rand.get()
      if is_valid(i):
        yield labels[i]

  return load_X(), load_y()


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
    
def split_set(files, d=0.8):
  split_test = int(d*len(files))
  return files[:split_test], files[split_test:]

def split(files):
  rest, test_files = split_set(files)
  train_files, validation_files = split_set(rest)
  return train_files, validation_files, test_files

def make_file_list(folder):
  return [(os.path.join(folder, label, filename), label)
    for label in os.listdir(folder)
    for filename in os.listdir(os.path.join(folder, label))
  ]

def input_pipeline_py(folder):
  return split(make_file_list(folder))


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
    path = saver.save(sess, save_to, global_step=step)
    print("Saved model to %s" % path)

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


def model(X, y):
  return fgo.FGO16().build(X, y, train=True)


def main_skflow(import_dir, export_dir, train_dir, batch_size, steps):
  classifier = fgo.FGOEstimator(model_fn=model,
                                n_classes=61,
                                batch_size=batch_size,
                                steps=0,
                                learning_rate=1e-2,
                                continue_training=True)
  classifier.fit(np.ones([1, 224, 224, 3], dtype=np.float32), np.ones([1,], dtype=np.float32))
  classifier.restore_variables(import_dir)
  X, y = load_data(make_file_list(DATA_FOLDER))
  classifier.steps = steps
  classifier.fit(X, y, logdir=train_dir)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('import_dir')
  parser.add_argument('export_dir')
  parser.add_argument('--train', default="/tmp/fgo16")
  parser.add_argument('--batch', default=64, type=int)
  parser.add_argument('--steps', default=100000, type=int)
  args = parser.parse_args()
  main_skflow(args.import_dir, args.export_dir, args.train, args.batch, args.steps)

