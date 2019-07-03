import csv
import argparse

from os.path import join

from helpers import process

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--dataset-split', default='train', choices=['train', 'test'])
  args = parser.parse_args()

  q1 = list()
  q2 = list()
  label = list()
  with open(join(args.data_dir, '%s.csv' % args.dataset_split), 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
      q1.append(process(row[0]))
      q2.append(process(row[1]))
      label.append(row[2])

  with open(join(args.data_dir, '%s_processed.csv' % args.dataset_split), 'w') as file:
    writer = csv.writer(file)
    for p1, p2, l in zip(q1, q2, label):
      writer.writerow([p1, p2, l])
