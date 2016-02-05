from pprint import pprint
import tensorflow as tf
import utils
import fgo


# class FGO16(fgo.Model):
#   def get_conv_filter(self, name):
#     return tf.get_variable("filter")

#   def get_bias(self, name):
#     return tf.get_variable("bias")

#   def get_fc_weight(self, name):
#     return tf.get_variables("weight")

#   def get_fc_weight_mod(self, name):
#     return tf.get_variable("weight")

#   def get_bias_mod(self, name):
#     return tf.get_variable("bias")



with open("fgo16.tfmodel", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)


images = tf.placeholder("float", [None, 224, 224, 3], name="images")
# m = FGO16()
# m.build(images)


var_names = ["fc8/weight:0", "fc8/bias:0"]
# variables = [
#   tf.Variable(tf.zeros((4096, 10)), name="import/fc8/weight"),
#   tf.Variable(tf.zeros((10,)), name="import/fc8/bias")
# ]

variables = tf.import_graph_def(graph_def, input_map={ "images": images }, return_elements=var_names, name="")
print "graph loaded from disk"

graph = tf.get_default_graph()
# for v in variables:
#   tf.add_to_collection(tf.GraphKeys.VARIABLES, v)
# pprint([op.name for op in graph.get_operations()])

cat = utils.load_image("cat.jpg")


saver = tf.train.Saver(var_list=variables)

with tf.Session() as sess:
  saver.restore(sess, "fgo16.ckpt")

  batch = cat.reshape((1, 224, 224, 3))
  assert batch.shape == (1, 224, 224, 3)

  feed_dict = { images: batch }

  prob_tensor = graph.get_tensor_by_name("prob:0")
  prob = sess.run(prob_tensor, feed_dict=feed_dict)

utils.print_prob(prob[0])
