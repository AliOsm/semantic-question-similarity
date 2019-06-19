import csv
import argparse

import numpy as np
import pickle as pkl

from os import remove
from os.path import join
from keras.models import load_model
from keras_self_attention import SeqWeightedAttention
from keras_ordered_neurons import ONLSTM

from helpers import load_embeddings_dict
from helpers import map_sentence, f1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--embeddings-type', default='elmo', choices=['elmo', 'bert'])
  parser.add_argument('--model-path', default='checkpoints/epoch100.h5')
  parser.add_argument('--output-type', default='binary', choices=['binary', 'probability'])
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

  data = list()
  with open(join(args.data_dir, 'test_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      print('Prepare Data: %s' % (idx + 1), end='\r')
      data.append((
        map_sentence(row[0], embeddings_dict),
        map_sentence(row[1], embeddings_dict),
        int(row[2])
      ))

  try:
    remove(join(args.data_dir, 'submit.csv'))
  except:
    pass

  with open(join(args.data_dir, 'submit.csv'), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['QuestionPairID', 'Prediction'])

    for idx, example in enumerate(data):
      print('Predicting Example: %s' % (idx + 1), end='\r')
      prediction = model.predict([[np.array(example[0])], [np.array(example[1])]]).squeeze()
      if args.output_type == 'binary':
        if prediction >= 0.5:
          writer.writerow([example[2], 1])
        else:
          writer.writerow([example[2], 0])
      else:
        writer.writerow([example[2], prediction])
