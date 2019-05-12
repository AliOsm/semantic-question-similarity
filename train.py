import csv
import random
random.seed(961)
import argparse

from os.path import join
from keras.models import Input, Model
from keras.layers import Concatenate, Embedding, Dropout, Dense, LSTM, CuDNNLSTM, Bidirectional
from keras.optimizers import Adam, Nadam, Adamax
from keras.callbacks import ModelCheckpoint, Callback
from keras_self_attention import SeqWeightedAttention

from helpers import *
from data_generator import DataGenerator

def build_model(embeddings_matrix, doc2vec_size, words_num, chars_num):
  # Inputs
  q1_sent_input = Input(shape=(doc2vec_size,))
  q2_sent_input = Input(shape=(doc2vec_size,))

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
    LSTM(units=128, dropout=args.dropout_rate, return_sequences=True, kernel_initializer='glorot_normal')
  )
  word_attention = SeqWeightedAttention()
  q1_word_lstm1 = word_attention(word_lstm1(q1_word_embedding))
  q2_word_lstm1 = word_attention(word_lstm1(q2_word_embedding))

  char_lstm1 = Bidirectional(
    LSTM(units=128, dropout=args.dropout_rate, return_sequences=True, kernel_initializer='glorot_normal')
  )
  char_attention = SeqWeightedAttention()
  q1_char_lstm1 = char_attention(char_lstm1(q1_char_embedding))
  q2_char_lstm1 = char_attention(char_lstm1(q2_char_embedding))

  # Dense
  sent_dense1 = Dense(units=256, activation='relu', kernel_initializer='glorot_normal')
  q1_sent_dense1 = Dropout(args.dropout_rate)(sent_dense1(q1_sent_input))
  q2_sent_dense1 = Dropout(args.dropout_rate)(sent_dense1(q2_sent_input))

  word_dense1 = Dense(units=256, activation='relu', kernel_initializer='glorot_normal')
  q1_word_dense1 = Dropout(args.dropout_rate)(word_dense1(q1_word_lstm1))
  q2_word_dense1 = Dropout(args.dropout_rate)(word_dense1(q2_word_lstm1))

  char_dense1 = Dense(units=256, activation='relu', kernel_initializer='glorot_normal')
  q1_char_dense1 = Dropout(args.dropout_rate)(char_dense1(q1_char_lstm1))
  q2_char_dense1 = Dropout(args.dropout_rate)(char_dense1(q2_char_lstm1))

  # Concatenate
  q1_concat = Concatenate()([q1_sent_dense1, q1_word_dense1, q1_char_dense1])
  q2_concat = Concatenate()([q2_sent_dense1, q2_word_dense1, q2_char_dense1])

  # Dense
  dense1 = Dense(units=384, activation='relu', kernel_initializer='glorot_normal')
  q1_dense1 = Dropout(args.dropout_rate)(dense1(q1_concat))
  q2_dense1 = Dropout(args.dropout_rate)(dense1(q2_concat))

  # Concatenate
  concat = Concatenate()([q1_dense1, q2_dense1])

  # Dense
  dense2 = Dropout(args.dropout_rate)(Dense(units=384, activation='relu', kernel_initializer='glorot_normal')(concat))
  dense3 = Dropout(args.dropout_rate)(Dense(units=192, activation='relu', kernel_initializer='glorot_normal')(dense2))
  dense4 = Dropout(args.dropout_rate)(Dense(units=96, activation='relu', kernel_initializer='glorot_normal')(dense3))

  # Predict
  output = Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal')(dense4)

  model = Model([q1_sent_input, q1_word_input, q1_char_input, q2_sent_input, q2_word_input, q2_char_input], output)

  model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', f1])
  model.summary()

  return model

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--doc2vec-dir', default='doc2vec_dir')
  parser.add_argument('--embeddings-dir', default='embeddings_dir')
  parser.add_argument('--dropout-rate', default=0.2, type=float)
  parser.add_argument('--epochs', default=100, type=int)
  parser.add_argument('--batch-size', default=256, type=int)
  args = parser.parse_args()

  doc2vec_model = load_doc2vec_model(join(args.doc2vec_dir, 'model'))
  embeddings_model, index2word, word2index, embeddings_matrix = load_fasttext_embedding(join(args.embeddings_dir, 'model'))
  char2index = load_characters_mapping(join(args.data_dir, 'characters.pkl'))

  data = list()
  sentences = set()
  with open(join(args.data_dir, 'train_processed_enlarged.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      if idx == 0: continue
      data.append((map_sentence(row[0], doc2vec_model, word2index, char2index), map_sentence(row[1], doc2vec_model, word2index, char2index), int(row[2])))
      data.append((map_sentence(row[1], doc2vec_model, word2index, char2index), map_sentence(row[0], doc2vec_model, word2index, char2index), int(row[2])))
      sentences.add(row[0])
      sentences.add(row[1])

  for sentence in sentences:
    data.append((map_sentence(sentence, doc2vec_model, word2index, char2index), map_sentence(sentence, doc2vec_model, word2index, char2index), 1))

  random.shuffle(data)
  dev = data[:2000]
  train = data[2000:]

  train_q1, train_q2, train_label = zip(*train)
  dev_q1, dev_q2, dev_label = zip(*dev)

  model = build_model(embeddings_matrix, len(doc2vec_model.infer_vector(['تجربة'])), len(word2index), len(char2index))

  train_gen = DataGenerator(train_q1, train_q2, train_label, args.batch_size, word2index['<PAD>'], char2index['<PAD>'])
  dev_gen = DataGenerator(dev_q1, dev_q2, dev_label, args.batch_size, word2index['<PAD>'], char2index['<PAD>'])

  checkpoint_path = 'checkpoints/epoch{epoch:02d}.ckpt'
  checkpoint_cb = ModelCheckpoint(checkpoint_path, verbose=0)

  model.fit_generator(generator=train_gen, validation_data=dev_gen, epochs=args.epochs, callbacks=[checkpoint_cb])
