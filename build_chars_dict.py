import csv
import argparse

import pickle as pkl

from os.path import join

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  args = parser.parse_args()

  chars = set()

  with open(join(args.data_dir, 'train_processed.csv'), 'r') as file:
  	reader = csv.reader(file)
  	for idx, row in enumerate(reader):
  		if idx == 0: continue
  		chars.update(set(list(row[0])))
  		chars.update(set(list(row[1])))

  with open(join(args.data_dir, 'test_processed.csv'), 'r') as file:
  	reader = csv.reader(file)
  	for idx, row in enumerate(reader):
  		if idx == 0: continue
  		chars.update(set(list(row[0])))
  		chars.update(set(list(row[1])))

  chars = { char:idx for idx, char in enumerate(chars) }

  with open(join(args.data_dir, 'characters.pkl'), 'wb') as file:
  	pkl.dump(chars, file)
