import csv
import argparse

import numpy as np

from os.path import join
from gensim.models import FastText
from keras.models import load_model

def load_fasttext_embedding(embedding_path):
  model = FastText.load(embedding_path)

  index2word = model.wv.index2word
  index2word.append('<PAD>')
  index2word.append('<SOS>')
  index2word.append('<EOS>')
  index2word.append('<UNK>')

  word2index = { word:idx for idx, word in enumerate(index2word) }

  embeddings_matrix = model.wv.vectors
  embeddings_matrix = np.concatenate((embeddings_matrix, np.ones((1, len(embeddings_matrix[0]))) * 1))
  embeddings_matrix = np.concatenate((embeddings_matrix, np.ones((1, len(embeddings_matrix[0]))) * 2))
  embeddings_matrix = np.concatenate((embeddings_matrix, np.ones((1, len(embeddings_matrix[0]))) * 3))
  embeddings_matrix = np.concatenate((embeddings_matrix, np.ones((1, len(embeddings_matrix[0]))) * 4))

  return model, index2word, word2index, embeddings_matrix

def map_sentence(sentence, word2index):
  ints = [word2index['<SOS>']]
  for word in sentence:
    ints.append(word2index[word])
  ints.append(word2index['<EOS>'])
  return ints

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--embeddings-dir', default='embeddings_dir')
  parser.add_argument('--model-path', default='checkpoints/epoch50.ckpt')
  args = parser.parse_args()

  embeddings_model, index2word, word2index, embeddings_matrix = load_fasttext_embedding(join(args.embeddings_dir, 'model'))
  model = load_model(args.model_path)

  data = list()
  with open(join(args.data_dir, 'test_processed.csv'), 'r') as file:
    reader = csv.reader(file)
    for idx, row in enumerate(reader):
      if idx == 0: continue
      data.append((map_sentence(row[0].split(), word2index), map_sentence(row[1].split(), word2index), int(row[2])))

  with open(join(args.data_dir, 'submit.csv'), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['QuestionPairID', 'Prediction'])

    for idx, example in enumerate(data):
      print(idx, end='\r')
      prediction = model.predict([[np.array(example[0])], [np.array(example[1])]]).squeeze()
      if prediction >= 0.5:
        writer.writerow([example[2], 1])
      else:
        writer.writerow([example[2], 0])
