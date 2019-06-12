import csv
import random
random.seed(961)
import argparse

from os.path import join
from keras.models import Input, Model, load_model
from keras.layers import Lambda, Subtract, Multiply, Concatenate, Embedding, Dropout
from keras.layers import Dense, GRU, CuDNNGRU, LSTM, CuDNNLSTM, Bidirectional
from keras.optimizers import Adam, Adamax, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras_self_attention import SeqWeightedAttention
from keras_ordered_neurons import ONLSTM

from helpers import load_embeddings_dict
from helpers import map_sentence, f1
from data_generator import DataGenerator

def build_model(embeddings_size):
  # Inputs
  q1_embeddings_input = Input(shape=(None, embeddings_size,), name='q1_word_embeddings')
  q2_embeddings_input = Input(shape=(None, embeddings_size,), name='q2_word_embeddings')

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
  q1_dense1 = word_lstm1(q1_embeddings_input)
  q2_dense1 = word_lstm1(q2_embeddings_input)

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
  q1_dense2 = word_attention(word_lstm2(q1_dense1))
  q2_dense2 = word_attention(word_lstm2(q2_dense1))

  # Concatenate
  subtract = Subtract()([q1_dense2, q2_dense2])
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

  model = Model([q1_embeddings_input, q2_embeddings_input], output)

  model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', f1])
  model.summary()

  return model

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--embeddings-type', default='elmo', choices=['elmo', 'bert'])
  parser.add_argument('--dropout-rate', default=0.2, type=float)
  parser.add_argument('--epochs', default=100, type=int)
  parser.add_argument('--batch-size', default=256, type=int)
  parser.add_argument('--initial-epoch', default=0, type=int)
  parser.add_argument('--dev-split', default=500, type=int)
  args = parser.parse_args()

  embeddings_dict = load_embeddings_dict(join(args.data_dir, '%s_dict.pkl' % args.embeddings_type))

  data = list()
  with open(join(args.data_dir, 'train_processed_enlarged.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      print('Prepare Data: %s' % (idx + 1), end='\r')
      data.append((
        map_sentence(row[0], embeddings_dict),
        map_sentence(row[1], embeddings_dict),
        int(row[2])
      ))

  random.shuffle(data)
  train = data[args.dev_split:]
  dev = data[:args.dev_split]

  train_q1, train_q2, train_label = zip(*train)
  dev_q1, dev_q2, dev_label = zip(*dev)

  if args.initial_epoch == 0:
    model = build_model(
      len(embeddings_dict[list(embeddings_dict)[0]][0]),
    )
  else:
    model = load_model(
      filepath='checkpoints/epoch%s.h5' % args.initial_epoch,
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
    args.batch_size
  )

  dev_gen = DataGenerator(
    dev_q1,
    dev_q2,
    dev_label,
    args.batch_size
  )

  checkpoint_cb = ModelCheckpoint(
    filepath='checkpoints/epoch{epoch:02d}.h5',
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
