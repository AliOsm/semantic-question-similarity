import csv
import argparse
import numpy as np

from os import walk
from os.path import join
from keras.models import load_model
from keras_self_attention import SeqWeightedAttention
from keras_ordered_neurons import ONLSTM

from helpers import f1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-in', '--folder-path')
  args = parser.parse_args()

  files = []
  for (dirpath, dirnames, filenames) in walk(args.folder_path):
    for file in filenames:
      if 'h5' in file:
        files.append(file)
  
  models = list()
  for file in files:
    print('Loading model %s' % file)
    model = load_model(
      filepath=join(args.folder_path, file),
      custom_objects={
        'f1': f1,
        'SeqWeightedAttention': SeqWeightedAttention,
        'ONLSTM': ONLSTM
      }
    )
    models.append(model)

  print('Start averaging process...')
  weights = [model.get_weights() for model in models]

  new_weights = list()
  for weights_list_tuple in zip(*weights):
    new_weights.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])

  avg_model = models[0]
  avg_model.set_weights(new_weights)

  print('Saving averaged model to: %s' % join(args.folder_path, 'averaged'))
  avg_model.save(join(args.folder_path, 'averaged.h5'))
