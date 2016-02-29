import os.path
from google.protobuf import text_format
import tensorflow as tf
import skflow
from pprint import pprint


VGG_MEAN = [103.939, 116.779, 123.68]
DROPOUTS = "dropouts"
LOSSES = "losses"


class Model():
  def get_conv_filter(self, name, shape, trainable):
    raise NotImplementedError

  def get_bias(self, name, shape, trainable):
    raise NotImplementedError

  def get_fc_weight(self, name, shape, trainable):
    raise NotImplementedError

  def _max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
      padding='SAME', name=name)

  def _conv_layer(self, bottom, name, shape_f, shape_out, trainable):
    shape_w = (shape_f[0], shape_f[1], bottom.get_shape()[-1].value, shape_out)

    with tf.variable_scope(name) as scope:
      filt = self.get_conv_filter(name, shape_w, trainable)
      conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

      conv_biases = self.get_bias_conv(name, (shape_out,), trainable)
      bias = tf.nn.bias_add(conv, conv_biases)

      relu = tf.nn.relu(bias)
      return relu

  def _fc_layer(self, bottom, name, shape_out, trainable, weight_decay):
    # with tf.variable_scope(name) as scope:
    shape = bottom.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
         dim *= d
    x = tf.reshape(bottom, [-1, dim])

    shape_w = (x.get_shape()[-1].value, shape_out)
    weights = self.get_fc_weight(name, shape_w, trainable, weight_decay)
    biases = self.get_bias_fc(name, (shape_out,), trainable)

    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    return fc

  def _fc_layer_mod(self, bottom, name, shape_out, trainable):
    # with tf.variable_scope(name) as scope:
    shape_w = (bottom.get_shape()[-1].value, shape_out)
    weights = self.get_fc_weight_mod(name, shape_w, trainable)
    biases = self.get_bias_mod(name, (shape_out,), trainable)

    fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)
    return fc


  # Input should be an rgb image [batch, height, width, 3]
  # values scaled [0, 1]
  def graph(self, X, n_classes, mod=True):
    # self.images = tf.placeholder("float", [None, 224, 224, 3], name="images")
    # rgb = self.images
    rgb_scaled = X * 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(3, 3, rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(3, [
      blue - VGG_MEAN[0],
      green - VGG_MEAN[1],
      red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    relu1_1 = self._conv_layer(bgr, "conv1_1", (3,3), 64, trainable=False)
    relu1_2 = self._conv_layer(relu1_1, "conv1_2", (3,3), 64, trainable=False)
    pool1 = self._max_pool(relu1_2, 'pool')

    relu2_1 = self._conv_layer(pool1, "conv2_1", (3,3), 128, trainable=False)
    relu2_2 = self._conv_layer(relu2_1, "conv2_2", (3,3), 128, trainable=False)
    pool2 = self._max_pool(relu2_2, 'pool')

    relu3_1 = self._conv_layer(pool2, "conv3_1", (3,3), 256, trainable=False)
    relu3_2 = self._conv_layer(relu3_1, "conv3_2", (3,3), 256, trainable=False)
    relu3_3 = self._conv_layer(relu3_2, "conv3_3", (3,3), 256, trainable=False)
    pool3 = self._max_pool(relu3_3, 'pool')

    relu4_1 = self._conv_layer(pool3, "conv4_1", (3,3), 512, trainable=False)
    relu4_2 = self._conv_layer(relu4_1, "conv4_2", (3,3), 512, trainable=False)
    relu4_3 = self._conv_layer(relu4_2, "conv4_3", (3,3), 512, trainable=False)
    pool4 = self._max_pool(relu4_3, 'pool')

    relu5_1 = self._conv_layer(pool4, "conv5_1", (3,3), 512, trainable=False)
    relu5_2 = self._conv_layer(relu5_1, "conv5_2", (3,3), 512, trainable=False)
    relu5_3 = self._conv_layer(relu5_2, "conv5_3", (3,3), 512, trainable=False)
    pool5 = self._max_pool(relu5_3, 'pool')

    with tf.variable_scope("fc6"):
      drop = dropout(0.5)
      fc6 = self._fc_layer(pool5, "fc6", 4096, trainable=True, weight_decay=5e-4)
      relu6 = tf.nn.dropout(tf.nn.relu(fc6), drop)

    with tf.variable_scope("fc7"):
      drop = dropout(0.5)
      fc7 = self._fc_layer(relu6, "fc7", 4096, trainable=True, weight_decay=5e-4)
      relu7 = tf.nn.dropout(tf.nn.relu(fc7), drop)

    with tf.variable_scope("fc8"):
      if mod:
        logits = self._fc_layer_mod(relu7, "fc8", n_classes, trainable=True)
      else:
        logits = self._fc_layer(relu7, "fc8", n_classes, trainable=True)

    return logits


class FGO16(Model):
  def get_conv_filter(self, name, shape, trainable):
    return tf.get_variable("filter", shape, initializer=tf.constant_initializer(), trainable=trainable)

  def get_bias_conv(self, name, shape, trainable):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(), trainable=trainable)

  def get_bias_fc(self, name, shape, trainable):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(), trainable=trainable)

  def get_fc_weight(self, name, shape, trainable, decay=None):
    if decay:
      return variable_with_weight_decay("weight", shape, tf.constant_initializer(), decay)
    else:
      return tf.get_variable("weight", shape, initializer=tf.constant_initializer(), trainable=trainable)

  def get_fc_weight_mod(self, name, shape, trainable, decay=None):
    return self.get_fc_weight(name, shape, trainable, decay)

  def get_bias_mod(self, name, shape, trainable):
    return self.get_bias_fc(name, shape, trainable)


def variable_with_weight_decay(name, shape, initializer, wd):
  var = tf.get_variable(name, shape, initializer=initializer)
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name="weight_loss")
    tf.add_to_collection(LOSSES, weight_decay)
  return var


def dropout(prob):
  drop = tf.get_variable("dropout", [], initializer=tf.constant_initializer(prob), trainable=False)
  tf.add_to_collection(DROPOUTS, drop)
  return drop


def inference(logits):
  return tf.nn.softmax(logits, name="prob")


def cost(logits, labels, batch_size):
  with tf.name_scope("train"):
    n_classes = logits.get_shape()[1].value
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated, [batch_size, n_classes], 1.0, 0.0)
    cross = tf.nn.softmax_cross_entropy_with_logits(logits, dense_labels)
    cross_mean = tf.reduce_mean(cross, name="loss")
    tf.add_to_collection(LOSSES, cross_mean)
    return tf.add_n(tf.get_collection(LOSSES), name="total_loss")


def training(loss):
  learning_rate = 0.5e-2
  momentum = 0.9
  with tf.name_scope("train"):
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    return optimizer.minimize(loss, global_step=global_step), global_step


def summaries(loss):
  tf.scalar_summary(loss.op.name, loss)
  return tf.merge_all_summaries()


def accuracy(prob, labels):
  with tf.name_scope("test"):
    in_top_k = tf.nn.in_top_k(prob, labels, 1)
    return tf.reduce_sum(tf.cast(in_top_k, tf.float32))


def no_dropouts():
  dropouts = tf.get_collection(DROPOUTS)
  return {drop: 1.0 for drop in dropouts}


def inputs():
  images = tf.placeholder("float", [None, 224, 224, 3], name="images")
  labels = tf.placeholder(tf.int32, [None], name="labels")
  return images, labels


def init(model, batch_size, n_classes):
  images, labels = inputs()
  logits = model.graph(images, n_classes)
  pprint([v.name for v in tf.all_variables()])
  prob = inference(logits)
  loss = cost(logits, labels, batch_size)
  return images, labels, prob, loss

