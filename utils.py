import skimage
import skimage.io
import skimage.transform
import numpy as np




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
