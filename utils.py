import skimage
import skimage.io
import skimage.transform
import numpy as np


synset = [l.strip() for l in open('fgo_synsets.txt')]


def load_collection(files):
  return skimage.io.ImageCollection(files, load_func=load_image)


# returns image of shape [224, 224, 3]
# [height, width, depth]
def process_image(img):
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  return resized_img



def load_image(path):
  img = skimage.io.imread(path)
  return process_image(img)


def load_labels(labels):
  n_feat, n_examples = len(synset), len(labels)
  l = np.zeros((n_examples, n_feat))
  syns = [s.split()[0] for s in synset]
  for i in range(n_examples):
    l[i, syns.index(labels[i])] = 1.0
  return l


# returns the top1 string
def print_prob(prob):
  #print prob
  print "prob shape", prob.shape
  pred = np.argsort(prob)[::-1]

  # Get top1 label
  top1 = synset[pred[0]]
  print "Top1: ", top1
  # Get top5 label
  top5 = [synset[pred[i]] for i in range(5)]
  print "Top5: ", top5
  return top1
