import os
os.environ["GLOG_minloglevel"] = "2"

from utils import *
import skimage
import caffe
import numpy as np
import math
import tensorflow as tf
import skflow

import fgo


def get_indices(synfile="synset.txt", idfile="fgo_synsets.txt"):
  synsets = [l.strip().split()[0] for l in open(synfile)]
  wanted = [l.strip().split()[0] for l in open(idfile)]
  indices = np.array([synsets.index(w) if w in synsets else -1 for w in wanted])
  known = np.where(indices >= 0)
  return indices, known


#caffe.set_mode_cpu()
net_caffe = caffe.Net("VGG_2014_16.prototxt", "VGG_ILSVRC_16_layers.caffemodel", caffe.TEST)


caffe_layers = {}
for i, layer in enumerate(net_caffe.layers):
    layer_name = net_caffe._layer_names[i]
    caffe_layers[layer_name] = layer

def caffe_weights(layer_name):
    layer = caffe_layers[layer_name]
    return layer.blobs[0].data

def caffe_bias(layer_name):
    layer = caffe_layers[layer_name]
    return layer.blobs[1].data

# converts caffe filter to tf
# tensorflow uses [filter_height, filter_width, in_channels, out_channels]
#                  2               3            1            0
# need to transpose channel axis in the weights
# caffe:  a convolution layer with 96 filters of 11 x 11 spatial dimension
# and 3 inputs the blob is 96 x 3 x 11 x 11
# caffe uses [out_channels, in_channels, filter_height, filter_width] 
#             0             1            2              3
def caffe2tf_filter(name):
  f = caffe_weights(name)
  return f.transpose((2, 3, 1, 0))

# caffe blobs are [ channel, height, width ]
# this returns  [ height, width, channel ]
def caffe2tf_conv_blob(name):
  blob = net_caffe.blobs[name].data[0]
  return blob.transpose((1, 2, 0))

def caffe2tf_1d_blob(name):
  blob = net_caffe.blobs[name].data[0]
  return blob


class ModelFromCaffe(fgo.Model):
  def get_conv_filter(self, name, shape):
    w = caffe2tf_filter(name)
    t = tf.Variable(np.array(w, dtype=np.float32), name="filter")
    # t = tf.constant(w, dtype=tf.float32, name="filter")
    # print("%s: %s" % (t.name, t.get_shape()))
    return t

  def get_bias_conv(self, name, shape):
    b = caffe_bias(name)
    t = tf.Variable(np.array(b, dtype=np.float32), name="bias")
    # t = tf.constant(b, dtype=tf.float32, name="bias")
    # print("%s: %s" % (t.name, t.get_shape()))
    return t

  def get_bias_fc(self, name, shape):
    b = caffe_bias(name)
    t = tf.Variable(np.array(b, dtype=np.float32), name="bias")
    # print("%s: %s" % (t.name, t.get_shape()))
    return t

  def get_fc_weight(self, name, shape):
    cw = caffe_weights(name)
    if name == "fc6":
      assert cw.shape == (4096, 25088)
      cw = cw.reshape((4096, 512, 7, 7)) 
      cw = cw.transpose((2, 3, 1, 0))
      cw = cw.reshape(25088, 4096)
    else:
      cw = cw.transpose((1, 0))

    t = tf.Variable(np.array(cw, dtype=np.float32), name="weight")
    # print("%s: %s" % (t.name, t.get_shape()))
    return t

  def get_fc_weight_mod(self, name, shape):
    cw = caffe_weights(name).transpose((1,0))
    indices, known = get_indices()
    W = np.array(np.random.randn(cw.shape[0], len(indices)), dtype=np.float32) * 1e-2
    for i in known:
      W[:, i] = cw[:, indices[i]]
    t = tf.Variable(W, name="weight")
    return t

  def get_bias_mod(self, name, shape):
    b = caffe_bias(name)
    indices, known = get_indices()
    B = np.array(np.zeros((len(indices),)), dtype=np.float32)
    B[known] = b[indices[known]]
    t = tf.Variable(B, name="bias")
    # print("%s: %s" % (t.name, t.get_shape()))
    return t


def main():
  images = tf.placeholder("float", [None, 224, 224, 3], name="images")
  m = ModelFromCaffe()
  m.build(images)

  # graph = tf.get_default_graph()
  # graph_def = graph.as_graph_def()
  # print "graph_def byte size", graph_def.ByteSize()
  # graph_def_s = graph_def.SerializeToString()

  # save_path = "fgo16.tfmodel"
  # with open(save_path, "wb") as f:
  #   f.write(graph_def_s)
  # print "saved model to %s" % save_path

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(init)
    path = saver.save(sess, "checkpoints/fgo16")
    print("saved variables to %s" % path)


def model(X, y):
  prediction, loss = ModelFromCaffe().build(X, y, train=False)
  return prediction, loss


def main_skflow():
  classifier = fgo.FGOEstimator(model_fn=model, n_classes=61, steps=0)
  classifier.fit(np.ones([1, 224, 224, 3], dtype=np.float32), np.ones([1,], dtype=np.float32))
  classifier.save_variables("fgo16-skflow")


if __name__ == "__main__":
  main_skflow()
