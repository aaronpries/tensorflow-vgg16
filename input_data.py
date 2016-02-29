import random
import os
import numpy as np
import skimage


synset = [l.strip() for l in open('fgo_synsets.txt')]


def load_collection(files, randflip, randshift, randcrop):
  return skimage.io.ImageCollection(files, load_func=load_image(randflip, randshift, randcrop))


def load_labels_indices(labels):
  syns = [s.split()[0] for s in synset]
  indices = np.array([syns.index(s) for s in labels])
  assert indices.shape == (len(labels),)
  assert (indices < len(syns)).all()
  assert (indices >= 0).all()
  return indices


# returns image of shape [224, 224, 3]
# [height, width, depth]
def crop(img, randomize):
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  if randomize:
    yy = random.randint(0, img.shape[0] - short_edge)
    xx = random.randint(0, img.shape[1] - short_edge)
  else:
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  return resized_img


def random_flip(im):
  if random.random() >= 0.5:
    flipped = np.fliplr(im)
    if len(im.shape) == 3:
      assert flipped[0,3,0] == im[0,-4,0]
    else:
      assert flipped[0,3] == im[0,-4]
    return flipped
  return im


def random_shift(im):
  if random.random() >= 0.5:
    return im
  return im


def load_image(randflip, randshift, randcrop):
  def load(path):
    img = skimage.io.imread(path)
    img = crop(img, randcrop)
    if randflip: img = random_flip(img)
    if randshift: img = random_shift(img)
    return img
  return load


def load_labels(labels):
  n_feat, n_examples = len(synset), len(labels)
  l = np.zeros((n_examples, n_feat))
  syns = [s.split()[0] for s in synset]
  for i in range(n_examples):
    l[i, syns.index(labels[i])] = 1.0
  return l


def load_batches(files, batch_size, finite=True, shuffle=True, randflip=False, randshift=False, randcrop=False):
  collection = load_collection([f for f,l in files], randflip, randshift, randcrop)
  labels = load_labels_indices([l for f,l in files])
  processed = 0
  idx = 0
  while not finite or processed < len(files):
    _images = []
    _labels = []
    i = 0
    while i < batch_size:
      if shuffle:
        idx = random.randint(0,len(files)-1)
      else:
        idx = (idx + 1) % len(files)
      try:
        x = collection[idx]
        if x.shape == (224,224,3):
          _images.append(x)
          _labels.append(labels[idx])
          i += 1
      except IOError: pass
    im = np.stack(_images, axis=0)
    lab = np.stack(_labels, axis=0)
    assert im.shape == (batch_size, 224, 224, 3)
    assert lab.shape == (batch_size,)
    yield im, lab
    processed += batch_size


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


def get_indices(synfile="synset.txt", idfile="fgo_synsets.txt"):
  synsets = [l.strip().split()[0] for l in open(synfile)]
  wanted = [l.strip().split()[0] for l in open(idfile)]
  indices = np.array([synsets.index(w) if w in synsets else -1 for w in wanted])
  known = np.where(indices >= 0)[0]
  return indices, known


def keep_known(files, synfile="synset.txt"):
  synsets = [l.strip().split()[0] for l in open(synfile)]
  indices, _ = get_indices()
  known = [synsets[i] for i in indices]
  return [(f,l) for f,l in files if l in known]

