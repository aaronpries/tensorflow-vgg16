import unittest

import fgo_train


class PreprocessingTest(unittest.TestCase):

  def test_split_set(self):
    set1, set2 = fgo_train.split_set(range(100))
    self.assertEqual(len(set1), 80)
    self.assertEqual(len(set2), 20)
    set1, set2 = fgo_train.split_set(range(100), 0.5)
    self.assertEqual(len(set1), 50)
    self.assertEqual(len(set2), 50)

  def test_split(self):
    train, validation, test = fgo_train.split(range(100))
    self.assertEqual(len(train), 64)
    self.assertEqual(len(validation), 16)
    self.assertEqual(len(test), 20)

  # def test_generator(self):
  #   train, _, _ = fgo_train.input_pipeline_py("test/imagenet")
  #   batches = fgo_train.batches(train, 3)
  #   first, labels = batches.next()
  #   self.assertEqual(len(first), 3)
  #   self.assertTupleEqual(first[0].shape, (224, 224, 3))
  #   self.assertTupleEqual(labels.shape, (3, 61))


  def test_Xy_generator(self):
    files = fgo_train.make_file_list("test/imagenet")
    print(files)
    X, y = fgo_train.load_data(files)
    self.assertNotEqual(X.next().mean(), X.next().mean())



if __name__ == '__main__':
      unittest.main()
