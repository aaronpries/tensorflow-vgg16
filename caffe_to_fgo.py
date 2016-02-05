import os
os.environ["GLOG_minloglevel"] = "2"

from utils import *
import matplotlib.pyplot as plt
import skimage
import caffe
import numpy as np
import tensorflow as tf
import fgo


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
  def get_conv_filter(self, name):
    w = caffe2tf_filter(name)
    return tf.constant(w, dtype=tf.float32, name="filter")

  def get_bias(self, name):
    b = caffe_bias(name)
    return tf.constant(b, dtype=tf.float32, name="bias")

  def get_fc_weight(self, name):
    cw = caffe_weights(name)
    if name == "fc6":
      assert cw.shape == (4096, 25088)
      cw = cw.reshape((4096, 512, 7, 7)) 
      cw = cw.transpose((2, 3, 1, 0))
      cw = cw.reshape(25088, 4096)
    else:
      cw = cw.transpose((1, 0))

    return tf.constant(cw, dtype=tf.float32, name="weight")

  def get_fc_weight_mod(self, name):
    cw = caffe_weights(name).transpose((1,0))
    W = cw[:, :10]
    # return tf.constant(W, dtype=tf.float32, name="weight")
    return tf.Variable(W, name="weight")

  def get_bias_mod(self, name):
    b = caffe_bias(name)
    B = b[:10]
    # return tf.constant(B, dtype=tf.float32, name="bias")
    return tf.Variable(B, name="bias")


def main():
  images = tf.placeholder("float", [None, 224, 224, 3], name="images")
  m = ModelFromCaffe()
  m.build(images)

  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()
  print "graph_def byte size", graph_def.ByteSize()
  graph_def_s = graph_def.SerializeToString()

  save_path = "fgo16.tfmodel"
  with open(save_path, "wb") as f:
    f.write(graph_def_s)

  print "saved model to %s" % save_path

  print([v.name for v in tf.all_variables()])
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(init)
    path = saver.save(sess, "fgo16.ckpt")
    print("saved variables to %s" % path)


if __name__ == "__main__":
  main()
