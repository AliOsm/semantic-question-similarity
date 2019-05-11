import csv
import argparse

from os.path import join
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--doc2vec-dir', default='doc2vec_dir')
  args = parser.parse_args()

  sentences = list()

  with open(join(args.data_dir, 'train_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      if idx == 0: continue
      sentences.append(row[0].split())
      sentences.append(row[1].split())

  with open(join(args.data_dir, 'test_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      if idx == 0: continue
      sentences.append(row[0].split())
      sentences.append(row[1].split())

  tagged_data = [TaggedDocument(words=sentence, tags=[str(i)]) for i, sentence in enumerate(sentences)]

  model = Doc2Vec(tagged_data, vector_size=512, window=5, min_count=1, epochs=50, workers=4, dm=1)
  model.save(join(args.doc2vec_dir, 'model'))
