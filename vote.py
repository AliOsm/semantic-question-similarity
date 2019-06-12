import csv
import argparse
 
from os import walk
from os.path import join

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-in', '--folder-path')
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
          predictions[int(row[0])].append(int(row[1]))
        except:
          predictions[int(row[0])] = [int(row[1])]

  with open(join(args.folder_path, 'vote.csv'), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['QuestionPairID', 'Prediction'])

    for example in predictions:
      ones = sum(predictions[example])
      zeros = 5 - ones

      if ones > zeros:
        writer.writerow([example, 1])
      else:
        writer.writerow([example, 0])
