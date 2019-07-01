import csv
import argparse

import numpy as np
import pickle as pkl

from os import remove
from os.path import join
from keras.models import load_model, Model
from keras_self_attention import SeqWeightedAttention
from keras_ordered_neurons import ONLSTM

from helpers import load_embeddings_dict
from helpers import map_sentence, f1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--embeddings-type', default='elmo', choices=['elmo', 'bert'])
  parser.add_argument('--model-path', default='sentences_embeddings_plots/plot-model.h5')
  args = parser.parse_args()

  embeddings_dict = load_embeddings_dict(join(args.data_dir, '%s_dict.pkl' % args.embeddings_type))

  model = load_model(
    filepath=args.model_path,
    custom_objects={
      'f1': f1,
      'SeqWeightedAttention': SeqWeightedAttention,
      'ONLSTM': ONLSTM
    }
  )

  visualization_model = Model(model.inputs[0], model.layers[4].get_output_at(0))
  visualization_model.summary()

  sentences = set()
  with open(join(args.data_dir, 'train_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      sentences.add(row[0])
      sentences.add(row[1])
  sentences = list(sentences)

  embeddings = list()
  for sentence in sentences:
    embeddings.append((sentence, visualization_model.predict(np.array([embeddings_dict[sentence]])).squeeze()))

  with open(join(args.data_dir, 'sentences_embeddings.pkl'), 'wb') as file:
    pkl.dump(embeddings, file)
