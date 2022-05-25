dataset = 'maxcut-instances/setB'

# Read all .txt files from folder dataset
import os

files = []
for file in os.listdir(dataset):
    if file.endswith(".txt"):
        files.append(file)


import networkx as nx
import matplotlib.pyplot as plt

for file in files:
    # Create graph from txt files with format 'node1 node2 weight'
    G = nx.Graph()
    with open(dataset + '/' + file, 'r') as f:
        # Skip first line
        next(f)
        for line in f:
            node1, node2, weight = line.split()
            G.add_edge(node1, node2, weight=float(weight))

    # Plot graph
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()