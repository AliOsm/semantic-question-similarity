import argparse
import arabic_reshaper
import pickle as pkl
import matplotlib.pyplot as plt

from os import sep, listdir
from os.path import isdir, join
from sklearn.manifold import TSNE
from bidi.algorithm import get_display

def tsne_plot(tokens, labels):
  tsne_model = TSNE(perplexity=20, random_state=1, n_components=2, init='pca')
  new_values = tsne_model.fit_transform(tokens)

  x = []
  y = []
  for value in new_values:
    x.append(value[0])
    y.append(value[1])

  plt.figure(figsize=(16, 8)) 
  for i in range(len(x)):
    plt.scatter(x[i], y[i])
    plt.annotate(get_display(arabic_reshaper.reshape(labels[i])),
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  args = parser.parse_args()

  with open(join(args.data_dir, 'sentences_embeddings.pkl'), 'rb') as file:
  	embeddings = pkl.load(file)
  sentences, embeddings = zip(*embeddings)

  tsne_plot(embeddings, sentences)
