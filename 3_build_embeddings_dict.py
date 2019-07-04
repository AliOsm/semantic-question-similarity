import csv
import json
import argparse

import numpy as np
import pickle as pkl

from os.path import join
from elmoformanylangs import Embedder
from bert_embedding import BertEmbedding

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--elmo-dir', default='elmo_dir')
  parser.add_argument('--batch-size', default=64, type=int)
  parser.add_argument('--embeddings-type', default='elmo', choices=['elmo', 'bert'])
  args = parser.parse_args()

  if args.embeddings_type == 'elmo':
    with open(join(args.elmo_dir, 'config.json'), 'r') as file:
      elmo_config_json = json.load(file)
    elmo_config_json['config_path'] = 'cnn_50_100_512_4096_sample.json'
    with open(join(args.elmo_dir, 'config.json'), 'w') as file:
      json.dump(elmo_config_json, file)

    elmo_model = Embedder(args.elmo_dir, batch_size=args.batch_size)
  else:
    bert_model = BertEmbedding(dataset_name='wiki_multilingual_cased')

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
  if args.embeddings_type == 'elmo':
    sentences = [sentence.split() for sentence in sentences]
  else:
    sentences = [sentence for sentence in sentences]

  if args.embeddings_type == 'elmo':
    features = elmo_model.sents2elmo(sentences)
    for idx in range(len(features)):
      features[idx] = np.array([[1] * len(features[0][0])] + list(features[idx]) + [[2] * len(features[0][0])])
  else:
    _, features = zip(*bert_model(sentences))
  
  assert(len(features) == len(sentences))

  embeddings_dict = dict()
  for sentence, feature in zip(sentences, features):
    if args.embeddings_type == 'elmo':
      embeddings_dict[' '.join(sentence)] = feature
    else:
      embeddings_dict[sentence] = feature

  assert(len(embeddings_dict) == len(sentences))

  with open(join(args.data_dir, '%s_dict.pkl' % args.embeddings_type), 'wb') as file:
    pkl.dump(embeddings_dict, file)
