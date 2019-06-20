import csv
import random
import argparse
 
from os import walk
from os.path import join

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-in', '--folder-path')
  parser.add_argument('--vote-type', default='binary', choices=['binary', 'probability'])
  args = parser.parse_args()

  files = []
  for (dirpath, dirnames, filenames) in walk(args.folder_path):
    for file in filenames:
      if 'csv' in file:
        files.append(file)
  
  predictions = dict()
  for file in files:
    with open(join(args.folder_path, file), 'r') as f:
      reader = csv.reader(f)

      next(reader)
      for row in reader:
        try:
          if args.vote_type == 'binary':
            predictions[int(row[0])].append(int(row[1]))
          else:
            predictions[int(row[0])].append(float(row[1]))
        except:
          if args.vote_type == 'binary':
            predictions[int(row[0])] = [int(row[1])]
          else:
            predictions[int(row[0])] = [float(row[1])]

  with open(join(args.folder_path, 'vote-%s.csv' % args.vote_type), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['QuestionPairID', 'Prediction'])

    for example in predictions:
      if args.vote_type == 'binary':
        ones = sum(predictions[example])
        zeros = 5 - ones

        if ones == zeros:
          writer.writerow([example, random.randint(0, 1)])
        elif ones > zeros:
          writer.writerow([example, 1])
        else:
          writer.writerow([example, 0])
      else:
        s = sum(predictions[example]) / 5
        if s >= 0.5:
          writer.writerow([example, 1])
        else:
          writer.writerow([example, 0])
