import csv
import argparse
import sys

from os.path import join

# increase recursion limit
sys.setrecursionlimit(15000)

# define variables
n = 0
map_sentence = {}
rev_map_sentence = {}
graph = []
vis = []
vector = []
new_data = set()

# dfs traversing
def dfs(u):
  vector.append(u)
  vis[u] = True
  for v in graph[1][u]:
    if not vis[v]:
      dfs(v)

# add item with sorted sentences
def add_item(my_list):
  new_data.add(tuple(list(sorted([my_list[0], my_list[1]])) + [my_list[2]]))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', default='data_dir')
  args = parser.parse_args()
  
  with open(join(args.data_dir, 'train_processed.csv'), 'r') as file:
    # read data
    print('Reading file..')
    reader = csv.reader(file)
    next(reader)

    # map sentences to unique IDs
    print('Mapping sentences..')
    for row in reader:
      for i in range(2):
        if row[i] not in map_sentence:
          map_sentence[row[i]] = n
          rev_map_sentence[n] = row[i]
          n += 1

  with open(join(args.data_dir, 'train_processed.csv'), 'r') as file:
    # read data
    print('Reading file..')
    reader = csv.reader(file)
    next(reader)

    # build the graph
    print('Building graph..')
    graph1 = []
    for i in range(n):
      graph1.append([])
    graph.append(graph1)

    graph1 = []
    for i in range(n):
      graph1.append([])
    graph.append(graph1)

    for row in reader:
      label = int(row[2])
      q1 = map_sentence[row[0]]
      q2 = map_sentence[row[1]]
      graph[label][q1].append(q2)
      graph[label][q2].append(q1)
      # add original data
      add_item(row)

    # find connected components
    print('Finding connected components..')
    vis = [False] * n
    for u in range(n):
      if not vis[u]:
        vector = []
        dfs(u)
        print('Node: %d, component size: %d'%(u, len(vector)))
        for i in range(len(vector)):
          for j in range(i+1, len(vector)):
            a = rev_map_sentence[vector[i]]
            b = rev_map_sentence[vector[j]]
            add_item([a, b, '1'])
            for k in graph[0][vector[j]]:
              c = rev_map_sentence[k]
              add_item([a, c, '0'])
            for k in graph[0][vector[i]]:
              c = rev_map_sentence[k]
              add_item([b, c, '0'])

  # write new data
  print('Total examples: %d'%len(new_data))
  with open(join(args.data_dir, 'train_processed_enlarged.csv'), 'w') as file:
    writer = csv.writer(file)
    for row in new_data:
      writer.writerow(list(row))