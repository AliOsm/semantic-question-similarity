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

def read_extra_data(data_dir):
  sentences = list()
  for subdir, dirs, files in walk(data_dir):
    for file in files:
      subsentences = list(map(
        str.strip,
        process(open(join(subdir, file), 'r', encoding='windows-1256').read()).split('\n')
      ))
      for subsentence in subsentences:
        if len(subsentence) == 0:
          continue
        sentences.append(subsentence.split())
  return sentences

def load_embeddings_dict(file_path):
  with open(file_path, 'rb') as file:
    embeddings_dict = pkl.load(file)
  return embeddings_dict

def load_characters_mapping(file_path):
  with open(file_path, 'rb') as file:
    characters_mapping = pkl.load(file)

  characters_mapping['<PAD>'] = len(characters_mapping)
  characters_mapping['<SOS>'] = len(characters_mapping)
  characters_mapping['<EOS>'] = len(characters_mapping)
  characters_mapping['<UNK>'] = len(characters_mapping)

  return characters_mapping

def map_sentence(sentence, doc2vec_model, embeddings_dict, char2index):
  char_ints = [char2index['<SOS>']]
  for char in sentence:
    char_ints.append(char2index[char])
  char_ints.append(char2index['<EOS>'])

  return doc2vec_model.infer_vector(sentence.split()), embeddings_dict[sentence], char_ints

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
