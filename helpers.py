import numpy as np
import pickle as pkl
import keras.backend as K

from os.path import join
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec

def load_doc2vec_model(model_path):
  model = Doc2Vec.load(model_path)
  return model

def load_fasttext_embedding(model_path):
  model = FastText.load(model_path)

  index2word = model.wv.index2word
  index2word.append('<PAD>')
  index2word.append('<SOS>')
  index2word.append('<EOS>')
  index2word.append('<UNK>')

  word2index = { word:idx for idx, word in enumerate(index2word) }

  embeddings_matrix = model.wv.vectors
  embeddings_matrix = np.concatenate((embeddings_matrix, np.ones((1, len(embeddings_matrix[0]))) * 1))
  embeddings_matrix = np.concatenate((embeddings_matrix, np.ones((1, len(embeddings_matrix[0]))) * 2))
  embeddings_matrix = np.concatenate((embeddings_matrix, np.ones((1, len(embeddings_matrix[0]))) * 3))
  embeddings_matrix = np.concatenate((embeddings_matrix, np.ones((1, len(embeddings_matrix[0]))) * 4))

  return model, index2word, word2index, embeddings_matrix

def load_characters_mapping(file_path):
  with open(file_path, 'rb') as file:
    characters_mapping = pkl.load(file)

  characters_mapping['<PAD>'] = len(characters_mapping)
  characters_mapping['<SOS>'] = len(characters_mapping)
  characters_mapping['<EOS>'] = len(characters_mapping)
  characters_mapping['<UNK>'] = len(characters_mapping)

  return characters_mapping

def map_sentence(sentence, doc2vec_model, word2index, char2index):
  word_ints = [word2index['<SOS>']]
  for word in sentence.split():
    word_ints.append(word2index[word])
  word_ints.append(word2index['<EOS>'])

  char_ints = [char2index['<SOS>']]
  for char in sentence:
    char_ints.append(char2index[char])
  char_ints.append(char2index['<EOS>'])

  return doc2vec_model.infer_vector(sentence.split()), word_ints, char_ints

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
  return 2*((precision*recall)/(precision+recall+K.epsilon()))
