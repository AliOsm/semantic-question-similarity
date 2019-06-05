import numpy as np
import pickle as pkl
import keras.backend as K

from os import walk
from os.path import join
from string import punctuation as punc_list
punc_list += '،؛؟`’‘”“'

def process(line):
  for punc in punc_list:
    line = line.replace(punc, ' %s ' % punc)
  line = ' '.join(line.split())
  return line

def load_embeddings_dict(file_path):
  with open(file_path, 'rb') as file:
    embeddings_dict = pkl.load(file)
  return embeddings_dict

def map_sentence(sentence, embeddings_dict):
  return embeddings_dict[sentence]

def f1(y_true, y_pred):
  def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

  def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

  precision = precision(y_true, y_pred)
  recall = recall(y_true, y_pred)
  return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
