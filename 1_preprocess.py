import csv
import argparse

from os.path import join
from string import punctuation as punc_list
punc_list += '،؛؟`’‘”“'

def process(line):
  for punc in punc_list:
    line = line.replace(punc, ' %s ' % punc)
  line = ' '.join(line.split())
  return line

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  parser.add_argument('--type', default='train', choices=['train', 'test'])
  args = parser.parse_args()

  q1 = list()
  q2 = list()
  label = list()
  with open(join(args.data_dir, '%s.csv' % args.type), 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
      q1.append(process(row[0]))
      q2.append(process(row[1]))
      label.append(row[2])

  with open(join(args.data_dir, '%s_processed.csv' % args.type), 'w') as file:
    writer = csv.writer(file)
    for p1, p2, l in zip(q1, q2, label):
      writer.writerow([p1, p2, l])
