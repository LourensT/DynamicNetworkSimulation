import os
import csv

# THIS NEEDS TO BE EXECUTED IN ENVIRONMENT WITH graph-tool
# i.e. using conda activate graph-tool
# or docker environment

import graph_tool as gt
import graph_tool.clustering as clust
import graph_tool.topology as topol
import graph_tool.centrality as central
import graph_tool.correlations as correl
import numpy as np
from datetime import datetime


def calc_static_features(name):
    start = datetime.now()

    print(datetime.now().strftime("%H:%M:%S"), f"calculating static features for {name}")
    G = gt.Graph()
    G.load(f"networks/{name}.graphml", fmt="graphml")

    row = [name] + list(degrees(G)) + list(clustering_and_triangles(G)) + [clust.global_clustering(G)[0], correl.assortativity(G, 'total')] + list(cores(G))
    add_to_csv(row)

def degrees(G):
    l = G.get_total_degrees([i for i in G.vertices()])
    stats = np.mean(l), np.std(l), min(l), max(l), *list(np.quantile(l, [.125, .25, .5, .75, .875]))
    return stats

def clustering_and_triangles(G):
    l = clust.local_clustering(G).get_array()
    stats = float(np.mean(l)), float(np.std(l)), min(l), max(l), *list(np.quantile(l, [.5, .6, .7, .8, .9]))
    
    d = G.get_total_degrees([i for i in G.vertices()])
    t = [0.5*c*d*(d-1) for c,d in zip(l,d)]
    stats2 = np.mean(t), np.std(t), min(t), max(t), *list(np.quantile(t, [.5, .6, .7, .8, .9]))
    return stats + stats2

def cores(G):
    l = topol.kcore_decomposition(G).get_array()
    stats = np.mean(l), np.std(l), min(l), max(l), *list(np.quantile(l, [.25, .5, .75]))
    return stats

def closeness(G):
    l = central.closeness(G).get_array()
    stats = np.mean(l), np.std(l), min(l), max(l), *list(np.quantile(l, [.25, .5, .75]))
    return stats
    
def add_to_csv(row):
    fp = "./statistics.csv"
    with open(fp, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row) 

if __name__ == '__main__':
    columns = ['avg deg', 'std deg', 'min deg', 'max deg', 'Q1 deg', 'Q2 deg', 'Q3 deg', 'Q4 deg', 'Q5 deg', 
               'avg clust', 'std clust', 'min clust', 'max clust', 'Q1 clust', 'Q2 clust', 'Q3 clust', 'Q4 clust', 'Q5 clust', 
               'avg tri', 'std tri', 'min tri', 'max tri', 'Q1 tri', 'Q2 tri', 'Q3 tri', 'Q4 tri', 'Q5 tri', 
               'transitivity',
               'assortativity',
               'avg core', 'std core', 'min core', 'max core', 'Q1 core', 'Q2 core', 'Q3 core'
              ]

    # check if static.csv exists, if not create it
    fp = "./statistics.csv"
 
    if not os.path.exists(fp):
        with open(fp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name",] + columns)

    # final list to be done
    names = [n.replace(".graphml", "") for n in  os.listdir("networks/") if '.graphml' in n]
    for name in names:
        calc_static_features(name)