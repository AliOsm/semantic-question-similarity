import csv
import argparse
import arabic_reshaper

import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from os.path import join
from keras.models import load_model, Model
from keras_self_attention import SeqWeightedAttention
from keras_ordered_neurons import ONLSTM
from bidi.algorithm import get_display

from helpers import load_embeddings_dict
from helpers import map_sentence, f1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--embeddings-type', default='elmo', choices=['elmo', 'bert'])
  parser.add_argument('--model-path', default='plots/plot_model.h5')
  args = parser.parse_args()

  sentences = [
    'ما هو تعريف المدير العام ؟',
    'من هو المدير العام ؟',
    'ما أجمل ما قيل بالموت ؟',
    'ما هو الموت ؟'
  ]

  embeddings_dict = load_embeddings_dict(join(args.data_dir, '%s_dict.pkl' % args.embeddings_type))

  model = load_model(
    filepath=args.model_path,
    custom_objects={
      'f1': f1,
      'SeqWeightedAttention': SeqWeightedAttention,
      'ONLSTM': ONLSTM
    }
  )

  onlstm_output = Model(model.inputs[0], model.layers[3].get_output_at(0))
  onlstm_output.summary()

  attention_layer = model.layers[4]
  attention_layer.return_attention = True

  fig, axes = plt.subplots(nrows=len(sentences), ncols=1, figsize=(10, 10))

  for idx, (ax, sentence) in enumerate(zip(axes, sentences)):
    attention_weights = K.eval(
      attention_layer(
        K.variable(onlstm_output.predict(np.array([embeddings_dict[sentence]])))
      )[-1]
    )

    attention_weights[0] = list(reversed(list(attention_weights[0])))

    sentence = list(reversed(sentence.split()))
    ax.set_xticklabels(
      ['', '<end>'] + [get_display(arabic_reshaper.reshape(word)) for word in sentence] + ['<start>'],
      fontdict={'fontsize': 22},
      rotation=0
    )

    ax.get_yaxis().set_visible(False)

    im = ax.matshow(attention_weights, cmap='viridis')

  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  fig.colorbar(im, cax=cbar_ax)

  plt.subplots_adjust(hspace=0)
  plt.show()
