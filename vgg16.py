import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class Model():
    def get_conv_filter(self, name, shape):
        raise NotImplementedError

    def get_bias(self, name, shape):
        raise NotImplementedError

    def get_fc_weight(self, name, shape):
        raise NotImplementedError

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _conv_layer(self, bottom, shape_filter, shape_out, name):
        shape_weights = (shape_filter[0], shape_filter[1], bottom.get_shape()[-1].value, shape_out)

        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name, shape_weights)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, (shape_out,))
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, bottom, shape_out, name):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            shape_w = (x.get_shape()[-1].value, shape_out)
            weights = self.get_fc_weight(name, shape_w)
            biases = self.get_bias(name, (shape_out,))

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def _build_conv(self, rgb, mean_value):
        # Build the convolutional (and pooling) layers of VGG
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]

        bgr = tf.concat(3, [
            blue - mean_value[0],
            green - mean_value[1],
            red - mean_value[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.relu1_1 = self._conv_layer(bgr, (3,3), 64, "conv1_1")
        self.relu1_2 = self._conv_layer(self.relu1_1, (3,3), 64, "conv1_2")
        self.pool1 = self._max_pool(self.relu1_2, 'pool1')

        self.relu2_1 = self._conv_layer(self.pool1, (3,3), 128, "conv2_1")
        self.relu2_2 = self._conv_layer(self.relu2_1, (3,3), 128, "conv2_2")
        self.pool2 = self._max_pool(self.relu2_2, 'pool2')

        self.relu3_1 = self._conv_layer(self.pool2, (3,3), 256, "conv3_1")
        self.relu3_2 = self._conv_layer(self.relu3_1, (3,3), 256, "conv3_2")
        self.relu3_3 = self._conv_layer(self.relu3_2, (3,3), 256, "conv3_3")
        self.pool3 = self._max_pool(self.relu3_3, 'pool3')

        self.relu4_1 = self._conv_layer(self.pool3, (3,3), 512, "conv4_1")
        self.relu4_2 = self._conv_layer(self.relu4_1, (3,3), 512, "conv4_2")
        self.relu4_3 = self._conv_layer(self.relu4_2, (3,3), 512, "conv4_3")
        self.pool4 = self._max_pool(self.relu4_3, 'pool4')

        self.relu5_1 = self._conv_layer(self.pool4, (3,3), 512, "conv5_1")
        self.relu5_2 = self._conv_layer(self.relu5_1, (3,3), 512, "conv5_2")
        self.relu5_3 = self._conv_layer(self.relu5_2, (3,3), 512, "conv5_3")
        self.pool5 = self._max_pool(self.relu5_3, 'pool5')

    def _build_fc(self, train):
        self.fc6 = self._fc_layer(self.pool5, 4096, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]

        self.relu6 = tf.nn.relu(self.fc6)
        if train:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self._fc_layer(self.relu6, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

    def _build_softmax(self):
        self.fc8 = self._fc_layer(self.relu7, 1000, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build(self, rgb, train=False, depth='softmax', mean_value=VGG_MEAN):
        assert isinstance(depth, str)

        if not isinstance(mean_value, list):
            mean_value = [mean_value]
        assert len(mean_value) in [1, 3]
        if len(mean_value) == 1:
            mean_value *= 3  # repeat the value 3 times

        self._build_conv(rgb, mean_value)
        if depth == 'conv5':
            return self.pool5

        if depth == 'fc7' or depth == 'softmax':
            self._build_fc(train)
            if depth == 'fc7':
                return self.relu7
            else:
                self._build_softmax()
                return self.prob


class TrainableModel(Model):
  def get_conv_filter(self, name, shape):
    return tf.get_variable("filter", shape, initializer=tf.constant_initializer())

  def get_bias(self, name, shape):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer())

  def get_fc_weight(self, name, shape):
    return tf.get_variable("weight", shape, initializer=tf.constant_initializer())
