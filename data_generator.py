import numpy as np

from keras.utils import Sequence

class DataGenerator(Sequence):
  def __init__(self, q1, q2, Y, batch_size, pad_index):
    self.q1 = q1
    self.q2 = q2
    self.Y = Y
    self.batch_size = batch_size
    self.pad_index = pad_index

  def __len__(self):
    return int(np.ceil(len(self.q1) / float(self.batch_size)))

  def __getitem__(self, idx):
    q1_batch = np.array(self.q1[idx * self.batch_size:(idx + 1) * self.batch_size])
    q2_batch = np.array(self.q2[idx * self.batch_size:(idx + 1) * self.batch_size])
    Y_batch = np.array(self.Y[idx * self.batch_size:(idx + 1) * self.batch_size])
    
    q1_msl = np.max([len(x) for x in q1_batch])
    q2_msl = np.max([len(x) for x in q2_batch])

    q1_new = list()
    for x in q1_batch:
      x_new = list(x)
      x_new.extend([self.pad_index] * (q1_msl - len(x_new)))
      q1_new.append(np.array(x_new))
    q1_batch = q1_new

    q2_new = list()
    for x in q2_batch:
      x_new = list(x)
      x_new.extend([self.pad_index] * (q2_msl - len(x_new)))
      q2_new.append(np.array(x_new))
    q2_batch = q2_new

    return [np.array(q1_batch), np.array(q2_batch)], np.array(Y_batch)