import numpy as np

from keras.utils import Sequence

class DataGenerator(Sequence):
  def __init__(self, q1, q2, Y, batch_size, char_pad_index):
    self.q1 = q1
    self.q2 = q2
    self.Y = Y
    self.batch_size = batch_size
    self.char_pad_index = char_pad_index

  def __len__(self):
    return int(np.ceil(len(self.q1) / float(self.batch_size)))

  def __getitem__(self, idx):
    q1_batch = np.array(self.q1[idx * self.batch_size:(idx + 1) * self.batch_size])
    q1_sent_batch, q1_elmo_batch, q1_char_batch = zip(*q1_batch)
    q2_batch = np.array(self.q2[idx * self.batch_size:(idx + 1) * self.batch_size])
    q2_sent_batch, q2_elmo_batch, q2_char_batch = zip(*q2_batch)
    Y_batch = np.array(self.Y[idx * self.batch_size:(idx + 1) * self.batch_size])
    
    q1_elmo_msl = np.max([len(x) for x in q1_elmo_batch])
    q1_char_msl = np.max([len(x) for x in q1_char_batch])
    q2_elmo_msl = np.max([len(x) for x in q2_elmo_batch])
    q2_char_msl = np.max([len(x) for x in q2_char_batch])

    q1_elmo_new = list()
    for x in q1_elmo_batch:
      x_new = list(x)
      x_new.extend([[0] * len(x[0])] * (q1_elmo_msl - len(x_new)))
      q1_elmo_new.append(np.array(x_new))
    q1_elmo_batch = q1_elmo_new

    q2_elmo_new = list()
    for x in q2_elmo_batch:
      x_new = list(x)
      x_new.extend([[0] * len(x[0])] * (q2_elmo_msl - len(x_new)))
      q2_elmo_new.append(np.array(x_new))
    q2_elmo_batch = q2_elmo_new

    q1_char_new = list()
    for x in q1_char_batch:
      x_new = list(x)
      x_new.extend([self.char_pad_index] * (q1_char_msl - len(x_new)))
      q1_char_new.append(np.array(x_new))
    q1_char_batch = q1_char_new

    q2_char_new = list()
    for x in q2_char_batch:
      x_new = list(x)
      x_new.extend([self.char_pad_index] * (q2_char_msl - len(x_new)))
      q2_char_new.append(np.array(x_new))
    q2_char_batch = q2_char_new

    return [np.array(q1_sent_batch), np.array(q1_elmo_batch), np.array(q1_char_batch),
            np.array(q2_sent_batch), np.array(q2_elmo_batch), np.array(q2_char_batch)], \
           np.array(Y_batch)
    