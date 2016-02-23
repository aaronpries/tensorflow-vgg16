import random
import os
import numpy as np
import skimage


synset = [l.strip() for l in open('fgo_synsets.txt')]


def load_collection(files):
  return skimage.io.ImageCollection(files, load_func=load_image)


def load_labels_indices(labels):
  syns = [s.split()[0] for s in synset]
  return np.array([syns.index(s) for s in labels])


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

# def process_image(val):
#   img = tf.image.decode_jpeg(val, channels=3)
#   return tf.image.convert_image_dtype(img, tf.double)


# def decode_image(filequeue):
#   reader = tf.FixedLengthReader()
#   fname, value = reader.read(filequeue)
#   image = process_image(value)
#   label = fname.split("/")[-2]
#   return  image, label


# def input_pipeline():
#   filenames = tf.train.match_filenames_once(os.path.join(DATA_FOLDER, "n*/*"))
#   filequeue = tf.train.string_input_producer(filenames, num_epochs=2)

#   image, label = decode_image(filequeue)

#   batch_size = 5
#   min_after_dequeue = 10
#   capacity = min_after_dequeue + 3 * batch_size

#   image_batch, label_batch = tf.train.shuffle_batch([image, label],
#     batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

#   return image_batch, label_batch


class MultiRandom:
  def __init__(self, N, seed=None, listeners=2):
    self.N = N
    self.listeners = listeners
    random.seed(seed)
    self.change()

  def change(self):
    self.x = random.randint(0, self.N-1)
    self.count = 0

  def get(self):
    self.count += 1
    x = self.x
    if self.count >= self.listeners:
      self.change()
    return x


def load(files):
  random.shuffle(files)
  image_files, label_files = zip(*files)
  images = load_collection(image_files)
  labels = load_labels_indices(label_files)

  rand = MultiRandom(len(files), listeners=2)

  def is_valid(i):
    try:
      im = images[i]
      return im.shape == (224,224,3)
    except IOError: return False
    
  def load_X():
    while True:
      i = rand.get()
      if is_valid(i):
        im = images[i]
        if len(im.shape) == 2:
          im = np.expand_dims(im, 2)
        yield im

  def load_y():
    while True:
      i = rand.get()
      if is_valid(i):
        yield labels[i]

  return load_X(), load_y()


def batches(files, batch_size):
  collection = load_collection([f for f,l in files])
  labels = load_labels([l for f,l in files])
  def maybe(i):
    try: return (collection[i], labels[i])
    except IOError: return None
  while True:
    sample = random.sample(range(len(files)), batch_size)
    l = [maybe(i) for i in sample]
    s = list(filter(lambda x: x is not None and x[0].shape == (224,224,3) and x[1].shape == (61,), l))
    im, lab = zip(*s)
    im = np.stack(im, axis=0)
    lab = np.stack(lab, axis=0)
    print(lab[0])
    yield im, lab
    

def split_set(files, d=0.8):
  split_test = int(d*len(files))
  return files[:split_test], files[split_test:]

def split(files):
  rest, test_files = split_set(files)
  train_files, validation_files = split_set(rest)
  return train_files, validation_files, test_files

def make_file_list(folder):
  return [(os.path.join(folder, label, filename), label)
    for label in os.listdir(folder)
    for filename in os.listdir(os.path.join(folder, label))
  ]

def make_split(folder):
  return split(make_file_list(folder))
