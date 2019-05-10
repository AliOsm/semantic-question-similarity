import csv
import random
random.seed(961)
import argparse

import numpy as np

from os.path import join
from gensim.models import FastText
from keras.models import Input, Model
from keras.layers import Concatenate, Embedding, Dropout, Dense, LSTM, CuDNNLSTM, Bidirectional
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint

from data_generator import DataGenerator

def load_fasttext_embedding(embedding_path):
  model = FastText.load(embedding_path)

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

def map_sentence(sentence, word2index):
  ints = [word2index['<SOS>']]
  for word in sentence:
    ints.append(word2index[word])
  ints.append(word2index['<EOS>'])
  return ints

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--embeddings-dir', default='embeddings_dir')
  parser.add_argument('--dropout-rate', default=0.2, type=float)
  parser.add_argument('--epochs', default=250, type=int)
  parser.add_argument('--batch-size', default=256, type=int)
  args = parser.parse_args()

  embeddings_model, index2word, word2index, embeddings_matrix = load_fasttext_embedding(join(args.embeddings_dir, 'model'))

  data = list()
  with open(join(args.data_dir, 'train_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      if idx == 0: continue
      data.append((map_sentence(row[0].split(), word2index), map_sentence(row[1].split(), word2index), int(row[2])))
      data.append((map_sentence(row[1].split(), word2index), map_sentence(row[0].split(), word2index), int(row[2])))

  random.shuffle(data)
  dev = data[:2000]
  train = data[2000:]

  train_q1, train_q2, train_label = zip(*train)
  dev_q1, dev_q2, dev_label = zip(*dev)

  q1_input = Input(shape=(None,))
  q2_input = Input(shape=(None,))

  embeddings = Embedding(input_dim=len(index2word),
                         output_dim=len(embeddings_matrix[0]),
                         weights=[embeddings_matrix],
                         trainable=True)
  q1_embedding = embeddings(q1_input)
  q2_embedding = embeddings(q2_input)

  lstm1 = Bidirectional(
    LSTM(units=128, dropout=args.dropout_rate, return_sequences=False, kernel_initializer='glorot_normal')
  )
  q1_lstm1 = lstm1(q1_embedding)
  q2_lstm1 = lstm1(q2_embedding)

  dense1 = Dense(units=256, activation='relu', kernel_initializer='glorot_normal')
  q1_dense1 = Dropout(args.dropout_rate)(dense1(q1_lstm1))
  q2_dense1 = Dropout(args.dropout_rate)(dense1(q2_lstm1))

  concat = Concatenate()([q1_dense1, q2_dense1])

  dense2 = Dropout(args.dropout_rate)(Dense(units=256, activation='relu', kernel_initializer='glorot_normal')(concat))
  dense3 = Dropout(args.dropout_rate)(Dense(units=128, activation='relu', kernel_initializer='glorot_normal')(dense2))

  output = Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal')(dense3)

  model = Model([q1_input, q2_input], output)

  model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
  model.summary()

  train_gen = DataGenerator(train_q1, train_q2, train_label, args.batch_size, word2index['<PAD>'])
  dev_gen = DataGenerator(dev_q1, dev_q2, dev_label, args.batch_size, word2index['<PAD>'])

  checkpoint_path = 'checkpoints/epoch{epoch:02d}.ckpt'
  checkpoint_cb = ModelCheckpoint(checkpoint_path, verbose=0)

  model.fit_generator(generator=train_gen, validation_data=dev_gen, epochs=args.epochs, callbacks=[checkpoint_cb])
