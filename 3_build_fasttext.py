import csv
import argparse

from os.path import join
from gensim.models import FastText

from helpers import read_watan, read_khaleej

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--fasttext-dir', default='fasttext_dir')
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

  sentences.extend(read_watan(args.data_dir))
  sentences.extend(read_khaleej(args.data_dir))

  print('Number of Sentences:', len(sentences))

  model = FastText(sentences, size=100, window=5, min_count=1, iter=5, workers=4, sg=1)
  print('Vocabulary Size:', len(model.wv.index2word))
  model.save(join(args.embeddings_dir, 'model'))

  print(model.most_similar('رجل'))
