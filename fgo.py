import tensorflow as tf

def load_graph(fname, var_names):
  with open(fname, mode='rb') as f:
    fileContent = f.read()
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(fileContent)
  images = tf.placeholder("float", [None, 224, 224, 3], name="images")
  variables = tf.import_graph_def(graph_def, input_map={ "images": images }, return_elements=var_names, name="")
  print("graph loaded from disk")
  return tf.get_default_graph(), variables


def load_graph_empty(train=False):
  model = FGO16()
  model.build(model.images, train)
  return model




VGG_MEAN = [103.939, 116.779, 123.68]

class Model():
  def __init__(self):
    self.images = tf.placeholder("float", [None, 224, 224, 3], name="images")

  def get_conv_filter(self, name, shape):
    raise NotImplementedError

  def get_bias(self, name, shape):
    raise NotImplementedError

  def get_fc_weight(self, name, shape):
    raise NotImplementedError

  def _max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
      padding='SAME', name=name)

  def _conv_layer(self, bottom, name, shape_w, shape_b):
    with tf.variable_scope(name) as scope:
      filt = self.get_conv_filter(name, shape_w)
      conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

      conv_biases = self.get_bias_conv(name, shape_b)
      bias = tf.nn.bias_add(conv, conv_biases)

      relu = tf.nn.relu(bias)
      return relu

  def _fc_layer(self, bottom, name, shape_w, shape_b):
    with tf.variable_scope(name) as scope:
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
             dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = self.get_fc_weight(name, shape_w)
        biases = self.get_bias_fc(name, shape_b)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc

  def _fc_layer_mod(self, bottom, name, shape_w, shape_b):
    with tf.variable_scope(name) as scope:
      shape = bottom.get_shape().as_list()
      dim = reduce(lambda acc,x: acc+x, shape[1:])
      x = tf.reshape(bottom, [-1, dim])

      weights = self.get_fc_weight_mod(name, shape_w)
      biases = self.get_bias_mod(name, shape_b)

      fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
      return fc


  # Input should be an rgb image [batch, height, width, 3]
  # values scaled [0, 1]
  def build(self, rgb, train=False):
    rgb_scaled = rgb * 255.0

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

    self.relu1_1 = self._conv_layer(bgr, "conv1_1", (3,3,3,64), (64,))
    self.relu1_2 = self._conv_layer(self.relu1_1, "conv1_2", (3,3,64,64), (64,))
    self.pool1 = self._max_pool(self.relu1_2, 'pool1')

    self.relu2_1 = self._conv_layer(self.pool1, "conv2_1", (3,3,64,128), (128,))
    self.relu2_2 = self._conv_layer(self.relu2_1, "conv2_2", (3,3,128,128), (128,))
    self.pool2 = self._max_pool(self.relu2_2, 'pool2')

    self.relu3_1 = self._conv_layer(self.pool2, "conv3_1", (3,3,128,256), (256,))
    self.relu3_2 = self._conv_layer(self.relu3_1, "conv3_2", (3,3,256,256), (256,))
    self.relu3_3 = self._conv_layer(self.relu3_2, "conv3_3", (3,3,256,256), (256,))
    self.pool3 = self._max_pool(self.relu3_3, 'pool3')

    self.relu4_1 = self._conv_layer(self.pool3, "conv4_1", (3,3,256,512), (512,))
    self.relu4_2 = self._conv_layer(self.relu4_1, "conv4_2", (3,3,512,512), (512,))
    self.relu4_3 = self._conv_layer(self.relu4_2, "conv4_3", (3,3,512,512), (512,))
    self.pool4 = self._max_pool(self.relu4_3, 'pool4')

    self.relu5_1 = self._conv_layer(self.pool4, "conv5_1", (3,3,512,512), (512,))
    self.relu5_2 = self._conv_layer(self.relu5_1, "conv5_2", (3,3,512,512), (512,))
    self.relu5_3 = self._conv_layer(self.relu5_2, "conv5_3", (3,3,512,512), (512,))
    self.pool5 = self._max_pool(self.relu5_3, 'pool5')

    self.fc6 = self._fc_layer(self.pool5, "fc6", (25088, 4096), (4096,))
    assert self.fc6.get_shape().as_list()[1:] == [4096]

    self.relu6 = tf.nn.relu(self.fc6)
    if train:
      self.relu6 = tf.nn.dropout(self.relu6, 0.5)

    self.fc7 = self._fc_layer(self.relu6, "fc7", (4096,4096), (4096,))
    self.relu7 = tf.nn.relu(self.fc7)
    if train:
      self.relu7 = tf.nn.dropout(self.relu7, 0.5)

    self.fc8 = self._fc_layer_mod(self.relu7, "fc8", (4096,61), (61,))
    self.prob = tf.nn.softmax(self.fc8, name="prob")

    self.global_step = tf.Variable(0, name="global_step", trainable=False)


  def build_train(self, batch_size, dim):
    self.labels = tf.placeholder(tf.float32, shape=(batch_size, dim), name="labels")
    cross = tf.nn.softmax_cross_entropy_with_logits(self.prob, self.labels)

    self.loss = tf.reduce_mean(cross)

    learning_rate = 1e-2
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    self.train = optimizer.minimize(self.loss, global_step=self.global_step)


  def build_summary(self):
    tf.scalar_summary(self.loss.op.name, self.loss)
    self.summary = tf.merge_all_summaries()


class FGO16(Model):
  def get_conv_filter(self, name, shape):
    return tf.Variable(tf.zeros(shape), name="filter", trainable=False)

  def get_bias_conv(self, name, shape):
    return tf.Variable(tf.zeros(shape), name="bias", trainable=False)

  def get_bias_fc(self, name, shape):
    return tf.Variable(tf.zeros(shape), name="bias")

  def get_fc_weight(self, name, shape):
    return tf.Variable(tf.zeros(shape), name="weight")

  def get_fc_weight_mod(self, name, shape):
    return tf.Variable(tf.zeros(shape), name="weight")

  def get_bias_mod(self, name, shape):
    return tf.Variable(tf.zeros(shape), name="bias")

