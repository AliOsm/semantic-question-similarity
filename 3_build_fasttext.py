import csv
import argparse

from os.path import join
from gensim.models import FastText

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--embeddings-dir', default='embeddings_dir')
  args = parser.parse_args()

  sentences = list()

  with open(join(args.data_dir, 'train_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      sentences.append(row[0].split())
      sentences.append(row[1].split())

  with open(join(args.data_dir, 'test_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      sentences.append(row[0].split())
      sentences.append(row[1].split())

  model = FastText(sentences, size=100, window=5, min_count=1, iter=50, workers=4, sg=1)
  model.save(join(args.embeddings_dir, 'model'))

  print(model.most_similar('رجل'))
