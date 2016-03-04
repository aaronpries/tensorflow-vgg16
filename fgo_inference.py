import argparse
import tensorflow as tf
import numpy as np
from pprint import pprint

import fgo
import input_data


class Inference(): pass


def run(paths, env):
  load = input_data.load_image()
  images = [load(p) for p in paths]
  images = np.stack(images, axis=0)

  feed_dict = {env.images_op: images}

  with env.sess.as_default():
    prob = env.sess.run([env.logits_op], feed_dict=feed_dict)[0]
  
  for p in prob:
    labels = input_data.get_top_labels(np.squeeze(p))
    pprint(labels)


def setup(param):
  env = Inference()

  env.images_op, _ = fgo.inputs()
  env.logits_op = fgo.FGO16().graph(env.images_op, n_classes=61, wd=5e-4)
  # prob_op = fgo.inference(logits_op)

  saver = tf.train.Saver()

  if not param.gpu:
    print("no gpu")
    config = tf.ConfigProto(device_count={'GPU': 0})
  else:
    print("with gpu")
    config = tf.ConfigProto()

  env.sess = tf.Session(config=config)
  with env.sess.as_default():
    tf.initialize_all_variables().run()
    saver.restore(env.sess, param.saved)

  return env


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--saved')
  parser.add_argument('--images', nargs="+")
  parser.add_argument('--gpu', default=1, type=int)
  param = parser.parse_args()
  env = setup(param)
  run(param.images, env)
