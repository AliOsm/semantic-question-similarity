import csv
import argparse

import numpy as np
import pickle as pkl

from os.path import join
from gensim.models import FastText
from keras.models import load_model
import keras.backend as K

def load_fasttext_embedding():
  model = FastText.load(join(args.embeddings_dir, 'model'))

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

def load_characters_mapping():
  with open(join(args.data_dir, 'characters.pkl'), 'rb') as file:
    characters_mapping = pkl.load(file)

  characters_mapping['<PAD>'] = len(characters_mapping)
  characters_mapping['<SOS>'] = len(characters_mapping)
  characters_mapping['<EOS>'] = len(characters_mapping)
  characters_mapping['<UNK>'] = len(characters_mapping)

  return characters_mapping

def map_sentence(sentence, word2index, char2index):
  word_ints = [word2index['<SOS>']]
  for word in sentence.split():
    word_ints.append(word2index[word])
  word_ints.append(word2index['<EOS>'])

  char_ints = [char2index['<SOS>']]
  for char in sentence:
    char_ints.append(char2index[char])
  char_ints.append(char2index['<EOS>'])

  return word_ints, char_ints

def f1(y_true, y_pred):
  def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

  def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

  precision = precision(y_true, y_pred)
  recall = recall(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--embeddings-dir', default='embeddings_dir')
  parser.add_argument('--model-path', default='checkpoints/epoch100.ckpt')
  args = parser.parse_args()

  embeddings_model, index2word, word2index, embeddings_matrix = load_fasttext_embedding()
  char2index = load_characters_mapping()

  model = load_model(args.model_path, custom_objects={'f1': f1})

  data = list()
  with open(join(args.data_dir, 'test_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      if idx == 0: continue
      data.append((map_sentence(row[0], word2index, char2index), map_sentence(row[1], word2index, char2index), int(row[2])))

  with open(join(args.data_dir, 'submit.csv'), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['QuestionPairID', 'Prediction'])

    for idx, example in enumerate(data):
      print(idx, end='\r')
      prediction = model.predict([[np.array(example[0][0])], [np.array(example[0][1])], [np.array(example[1][0])], [np.array(example[1][1])]]).squeeze()
      if prediction >= 0.5:
        writer.writerow([example[2], 1])
      else:
        writer.writerow([example[2], 0])
