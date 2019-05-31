import csv
import argparse

import pickle as pkl

from os.path import join
from elmoformanylangs import Embedder

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--elmo-dir', default='elmo_dir')
  parser.add_argument('--batch-size', default=64, type=int)
  args = parser.parse_args()

  elmo_model = Embedder(args.elmo_dir, batch_size=args.batch_size)

  sentences = list()

  with open(join(args.data_dir, 'train_processed_enlarged.csv'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      sentences.append(row[0])
      sentences.append(row[1])

  with open(join(args.data_dir, 'test_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      sentences.append(row[0])
      sentences.append(row[1])

  sentences = list(set(sentences))
  sentences = [sentence.split() for sentence in sentences]

  features = elmo_model.sents2elmo(sentences)
  
  assert(len(features) == len(sentences))

  elmo_dict = dict()
  for sentence, feature in zip(sentences, features):
    elmo_dict[' '.join(sentence)] = feature

  assert(len(elmo_dict) == len(sentences))

  with open(join(args.data_dir, 'elmo_dict.pkl'), 'wb') as file:
    pkl.dump(elmo_dict, file)
