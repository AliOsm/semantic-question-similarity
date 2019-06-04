import csv
import random
random.seed(961)
import argparse

from os.path import join
from gensim.models.doc2vec import Doc2Vec
from keras.models import Input, Model, load_model
from keras.layers import Lambda, Subtract, Multiply, Concatenate, Embedding, Dropout
from keras.layers import Dense, GRU, CuDNNGRU, LSTM, CuDNNLSTM, Bidirectional
from keras.optimizers import Adam, Adamax, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras_ordered_neurons import ONLSTM
from keras_self_attention import SeqWeightedAttention
from optimizer import NormalizedOptimizer

from helpers import load_elmo_dict, load_characters_mapping
from helpers import map_sentence, f1
from data_generator import DataGenerator

def build_model(doc2vec_size, elmo_size, chars_num):
  # Inputs
  q1_sent_input = Input(shape=(doc2vec_size,))
  q2_sent_input = Input(shape=(doc2vec_size,))

  q1_elmo_input = Input(shape=(None, elmo_size,))
  q2_elmo_input = Input(shape=(None, elmo_size,))

  q1_char_input = Input(shape=(None,))
  q2_char_input = Input(shape=(None,))

  # Embeddings
  char_embeddings = Embedding(input_dim=chars_num,
                              output_dim=25,
                              trainable=True,
                              embeddings_initializer='glorot_normal')
  q1_char_embedding = char_embeddings(q1_char_input)
  q2_char_embedding = char_embeddings(q2_char_input)

  # LSTM
  word_lstm1 = Bidirectional(
    ONLSTM(
      units=256,
      chunk_size=8,
      dropout=args.dropout_rate,
      return_sequences=True,
      kernel_initializer='glorot_normal'
    )
  )
  q1_word_lstm1 = word_lstm1(q1_elmo_input)
  q2_word_lstm1 = word_lstm1(q2_elmo_input)

  word_lstm2 = Bidirectional(
    ONLSTM(
      units=256,
      chunk_size=8,
      dropout=args.dropout_rate,
      return_sequences=True,
      kernel_initializer='glorot_normal'
    )
  )
  word_attention = SeqWeightedAttention()
  q1_word_lstm2 = word_attention(word_lstm2(q1_word_lstm1))
  q2_word_lstm2 = word_attention(word_lstm2(q2_word_lstm1))

  char_lstm1 = Bidirectional(
    ONLSTM(
      units=128,
      chunk_size=16,
      dropout=args.dropout_rate,
      return_sequences=True,
      kernel_initializer='glorot_normal'
    )
  )
  char_attention = SeqWeightedAttention()
  q1_char_lstm1 = char_attention(char_lstm1(q1_char_embedding))
  q2_char_lstm1 = char_attention(char_lstm1(q2_char_embedding))

  # Dense
  sent_dense1 = Dense(units=256, activation='relu', kernel_initializer='glorot_normal')
  q1_sent_dense1 = Dropout(args.dropout_rate)(sent_dense1(q1_sent_input))
  q2_sent_dense1 = Dropout(args.dropout_rate)(sent_dense1(q2_sent_input))

  # Concatenate
  q1_concat = Concatenate()([q1_sent_dense1, q1_word_lstm2, q1_char_lstm1])
  q2_concat = Concatenate()([q2_sent_dense1, q2_word_lstm2, q2_char_lstm1])

  # Concatenate
  subtract = Subtract()([q1_concat, q2_concat])
  multiply_subtract = Multiply()([subtract, subtract])
  
  # Dense
  dense1 = Dropout(args.dropout_rate)(
    Dense(units=1024, activation='relu', kernel_initializer='glorot_normal')(multiply_subtract)
  )
  dense2 = Dropout(args.dropout_rate)(
    Dense(units=512, activation='relu', kernel_initializer='glorot_normal')(dense1)
  )
  dense3 = Dropout(args.dropout_rate)(
    Dense(units=256, activation='relu', kernel_initializer='glorot_normal')(dense2)
  )
  dense4 = Dropout(args.dropout_rate)(
    Dense(units=128, activation='relu', kernel_initializer='glorot_normal')(dense3)
  )

  # Predict
  output = Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal')(dense4)

  model = Model([
    q1_sent_input, q1_elmo_input, q1_char_input,
    q2_sent_input, q2_elmo_input, q2_char_input
  ], output)

  model.compile(
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', f1]
  )
  model.summary()

  return model

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--doc2vec-dir', default='doc2vec_dir')
  parser.add_argument('--dropout-rate', default=0.2, type=float)
  parser.add_argument('--epochs', default=100, type=int)
  parser.add_argument('--batch-size', default=256, type=int)
  parser.add_argument('--initial-epoch', default=0, type=int)
  args = parser.parse_args()

  char2index = load_characters_mapping(join(args.data_dir, 'characters.pkl'))
  elmo_dict = load_elmo_dict(join(args.data_dir, 'elmo_dict.pkl'))
  doc2vec_model = Doc2Vec.load(join(args.doc2vec_dir, 'model'))

  data = list()
  sentences = set()
  cnt = 0
  with open(join(args.data_dir, 'train_processed_enlarged.csv'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      print('Prepare Data: %s' % (cnt), end='\r'); cnt += 2
      data.append((
        map_sentence(row[0], doc2vec_model, elmo_dict, char2index),
        map_sentence(row[1], doc2vec_model, elmo_dict, char2index),
        int(row[2])
      ))
      data.append((
        map_sentence(row[1], doc2vec_model, elmo_dict, char2index),
        map_sentence(row[0], doc2vec_model, elmo_dict, char2index),
        int(row[2])
      ))
      sentences.add(row[0])
      sentences.add(row[1])

  for sentence in sentences:
    print('Prepare Data: %s' % (cnt), end='\r'); cnt += 1
    data.append((
      map_sentence(sentence, doc2vec_model, elmo_dict, char2index),
      map_sentence(sentence, doc2vec_model, elmo_dict, char2index),
      1
    ))
  print('Prepare Data: Done')

  random.shuffle(data)
  train = data[2000:]
  dev = data[:2000]

  train_q1, train_q2, train_label = zip(*train)
  dev_q1, dev_q2, dev_label = zip(*dev)

  if args.initial_epoch == 0:
    model = build_model(
      len(doc2vec_model.infer_vector(['تجربة'])),
      len(elmo_dict[list(elmo_dict)[0]][0]),
      len(char2index)
    )
  else:
    model = load_model(
      filepath='checkpoints/epoch%s.ckpt' % args.initial_epoch,
      custom_objects={
        'f1': f1,
        'SeqWeightedAttention': SeqWeightedAttention,
        'ONLSTM': ONLSTM
      }
    )

  train_gen = DataGenerator(
    train_q1,
    train_q2,
    train_label,
    args.batch_size,
    char2index['<PAD>']
  )

  dev_gen = DataGenerator(
    dev_q1,
    dev_q2,
    dev_label,
    args.batch_size,
    char2index['<PAD>']
  )

  checkpoint_cb = ModelCheckpoint(
    filepath='checkpoints/epoch{epoch:02d}.ckpt',
    monitor='val_f1',
    verbose=1,
    save_best_only=False,
    mode='max',
    period=10
  )

  plateau_cb = ReduceLROnPlateau(
    monitor='val_f1',
    mode='max',
    factor=0.1,
    patience=5,
    verbose=1
  )

  model.fit_generator(generator=train_gen,
                      validation_data=dev_gen,
                      epochs=args.epochs,
                      callbacks=[checkpoint_cb],
                      initial_epoch=args.initial_epoch)
