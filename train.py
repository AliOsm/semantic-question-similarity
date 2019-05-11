import csv
import random
random.seed(961)
import argparse

import numpy as np
import pickle as pkl
import keras.backend as K

from os.path import join
from gensim.models import FastText
from keras.models import Input, Model
from keras.layers import Concatenate, Embedding, Dropout, Dense, LSTM, CuDNNLSTM, Bidirectional
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint

from data_generator import DataGenerator

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

def build_model(embeddings_matrix, words_num, chars_num):
  # Inputs
  q1_word_input = Input(shape=(None,))
  q2_word_input = Input(shape=(None,))

  q1_char_input = Input(shape=(None,))
  q2_char_input = Input(shape=(None,))

  # Embeddings
  word_embeddings = Embedding(input_dim=words_num,
                              output_dim=len(embeddings_matrix[0]),
                              weights=[embeddings_matrix],
                              trainable=True)
  q1_word_embedding = word_embeddings(q1_word_input)
  q2_word_embedding = word_embeddings(q2_word_input)

  char_embeddings = Embedding(input_dim=chars_num,
                              output_dim=25,
                              trainable=True)
  q1_char_embedding = char_embeddings(q1_char_input)
  q2_char_embedding = char_embeddings(q2_char_input)

  # LSTM
  word_lstm1 = Bidirectional(
    LSTM(units=128, dropout=args.dropout_rate, return_sequences=False, kernel_initializer='glorot_normal')
  )
  q1_word_lstm1 = word_lstm1(q1_word_embedding)
  q2_word_lstm1 = word_lstm1(q2_word_embedding)

  char_lstm1 = Bidirectional(
    LSTM(units=128, dropout=args.dropout_rate, return_sequences=False, kernel_initializer='glorot_normal')
  )
  q1_char_lstm1 = char_lstm1(q1_char_embedding)
  q2_char_lstm1 = char_lstm1(q2_char_embedding)

  # Dense
  word_dense1 = Dense(units=256, activation='relu', kernel_initializer='glorot_normal')
  q1_word_dense1 = Dropout(args.dropout_rate)(word_dense1(q1_word_lstm1))
  q2_word_dense1 = Dropout(args.dropout_rate)(word_dense1(q2_word_lstm1))

  char_dense1 = Dense(units=256, activation='relu', kernel_initializer='glorot_normal')
  q1_char_dense1 = Dropout(args.dropout_rate)(char_dense1(q1_char_lstm1))
  q2_char_dense1 = Dropout(args.dropout_rate)(char_dense1(q2_char_lstm1))

  # Concatenate
  q1_concat = Concatenate()([q1_word_dense1, q1_char_dense1])
  q2_concat = Concatenate()([q2_word_dense1, q2_char_dense1])

  # Dense
  dense1 = Dense(units=256, activation='relu', kernel_initializer='glorot_normal')
  q1_dense1 = Dropout(args.dropout_rate)(dense1(q1_concat))
  q2_dense1 = Dropout(args.dropout_rate)(dense1(q2_concat))

  # Concatenate
  concat = Concatenate()([q1_dense1, q2_dense1])

  # Dense
  dense2 = Dropout(args.dropout_rate)(Dense(units=256, activation='relu', kernel_initializer='glorot_normal')(concat))
  dense3 = Dropout(args.dropout_rate)(Dense(units=128, activation='relu', kernel_initializer='glorot_normal')(dense2))

  # Predict
  output = Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal')(dense3)

  model = Model([q1_word_input, q1_char_input, q2_word_input, q2_char_input], output)

  model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', f1])
  model.summary()

  return model

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--embeddings-dir', default='embeddings_dir')
  parser.add_argument('--dropout-rate', default=0.2, type=float)
  parser.add_argument('--epochs', default=100, type=int)
  parser.add_argument('--batch-size', default=256, type=int)
  args = parser.parse_args()

  embeddings_model, index2word, word2index, embeddings_matrix = load_fasttext_embedding()
  char2index = load_characters_mapping()

  data = list()
  sentences = set()
  cnt = [0, 0]
  with open(join(args.data_dir, 'train_processed_enlarged.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      if idx == 0: continue
      data.append((map_sentence(row[0], word2index, char2index), map_sentence(row[1], word2index, char2index), int(row[2])))
      data.append((map_sentence(row[1], word2index, char2index), map_sentence(row[0], word2index, char2index), int(row[2])))
      sentences.add(row[0])
      sentences.add(row[1])
      cnt[int(row[2])] += 2

  for sentence in sentences:
    data.append((map_sentence(sentence, word2index, char2index), map_sentence(sentence, word2index, char2index), 1))
    cnt[1] += 1

  print(len(data), cnt)

  random.shuffle(data)
  dev = data[:2000]
  train = data[2000:]

  train_q1, train_q2, train_label = zip(*train)
  dev_q1, dev_q2, dev_label = zip(*dev)

  model = build_model(embeddings_matrix, len(word2index), len(char2index))

  train_gen = DataGenerator(train_q1, train_q2, train_label, args.batch_size, word2index['<PAD>'], char2index['<PAD>'])
  dev_gen = DataGenerator(dev_q1, dev_q2, dev_label, args.batch_size, word2index['<PAD>'], char2index['<PAD>'])

  checkpoint_path = 'checkpoints/epoch{epoch:02d}.ckpt'
  checkpoint_cb = ModelCheckpoint(checkpoint_path, verbose=0)

  model.fit_generator(generator=train_gen, validation_data=dev_gen, epochs=args.epochs, callbacks=[checkpoint_cb])
