import csv
import random
import argparse

from os.path import join
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  args = parser.parse_args()

  random.seed(961)

  data = list()

  with open(join(args.data_dir, 'train_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      data.append(row)

  random.shuffle(data)
  train = data[1000:]
  dev = data[:1000]

  with open(join(args.data_dir, 'train_processed.csv'), 'w') as file:
    writer = csv.writer(file)

    for example in train:
      writer.writerow(example)

  with open(join(args.data_dir, 'dev_processed.csv'), 'w') as file:
    writer = csv.writer(file)

    for example in dev:
      writer.writerow(example)
