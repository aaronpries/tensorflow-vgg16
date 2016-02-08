from pprint import pprint
import tensorflow as tf
import utils
import sys

import fgo


with open("fgo16.tfmodel", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)


images = tf.placeholder("float", [None, 224, 224, 3], name="images")


var_names = ["fc8/weight:0", "fc8/bias:0"]

variables = tf.import_graph_def(graph_def, input_map={ "images": images }, return_elements=var_names, name="")
print("graph loaded from disk")

graph = tf.get_default_graph()

cat = utils.load_image(sys.argv[1])


saver = tf.train.Saver(var_list=variables)

with tf.Session() as sess:
  saver.restore(sess, "fgo16.ckpt")

  batch = cat.reshape((1, 224, 224, 3))
  assert batch.shape == (1, 224, 224, 3)

  feed_dict = { images: batch }

  prob_tensor = graph.get_tensor_by_name("prob:0")
  prob = sess.run(prob_tensor, feed_dict=feed_dict)

utils.print_prob(prob[0])
