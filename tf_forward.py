import tensorflow as tf
import utils
import vgg16


images = tf.placeholder("float", [None, 224, 224, 3], name='images')
output_tensor = vgg16.TrainableModel().build(images, train=False, depth='fc7')

cat = utils.load_image("cat.jpg")

with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)

  saver = tf.train.Saver()
  saver.restore(sess, 'vgg16_trainable.ckpt')

  batch = cat.reshape((1, 224, 224, 3))
  assert batch.shape == (1, 224, 224, 3)

  feed_dict = { images: batch }

  prob = sess.run(output_tensor, feed_dict={ images: batch })

utils.print_prob(prob[0])