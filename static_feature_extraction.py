# %%
import networkx as nx
import os
import csv
from datetime import datetime

from multiprocessing import Pool

from dynamic_feature_extraction import GraphAnalysis

import numpy as np
import pandas as pd

""" Static features feasible for classifying medium-to-large networks. Feature not repeated if already used by earlier paper.
            NB: scale invariance not needed, since training networks have same #nodes and approx. same #edges. Density is pointless, since constant.

    -- global clust. coef. (=transitivity), degree assortativity, 4-motif counts
            (Characterizing the structural diversity of complex networks across domains, 2017)
    -- avg local clust. coef., modularity, DDQC of degree distr.
            (Classification of complex networks based on similarity of topological network features, 2017)
    -- diameter, effective diameter, power-law exponent, discretized distr. of: degree, local clust. coef., core, betweenness, closeness, Katz, PageRank centralities. 
            (Towards a Systematic Evaluation of Generative Network Models, 2018)
    -- total/avg/max triangles centered on edge, max k-core number, max clique
            (Complex networks are structurally distinguishable by domain, 2019, 
             peer-reviewed later version of retracted paper from 2018)
    -- entropy of degree distr., #communities, 3-node and 4-node motif counts.
            (Empirically Classifying Network Mechanisms, 2021)
"""

def calc_static_features(id):
    start = datetime.now()

    print(datetime.now().strftime("%H:%M:%S"), f"calculating static features for {id}")
    g = GraphAnalysis(id, fp=os.path.join(os.getcwd(), "../", "Dataset", "DatasetNetworkSource"))
    G = g.get_graph_representation()

    # assert no self edges 
    selfloops = list(nx.selfloop_edges(G))
    assert len(selfloops) == 0, f"there are self loops! {selfloops, id}"

    # these don't scale:
    # print("\tpagerank distr.:", pageranks(G)) # doesn't scale
    # print("\tcloseness distr.:", closeness(G)) # doesn't scale
    # print("\tdiameter:", nx.diameter(G)) # doesn't scale


    row = [id] + list(degrees(G)) + list(clustering(G)) + list(triangles(G)) + [nx.transitivity(G), nx.degree_assortativity_coefficient(G)] + list(cores(G))
    add_to_csv(row)
    duration = (datetime.now() - start)
    print(f"network {id} done in {str(duration)}")

def degrees(G):
    deg_tuples = list(nx.degree(G)) # [(0, 5), (1, 2), ...]
    l = list(zip(*deg_tuples))[1]
    stats = np.mean(l), np.std(l), min(l), max(l), *list(np.quantile(l, [.125, .25, .5, .75, .875]))
    return stats

def clustering(G):
    l = list(nx.clustering(G).values())
    stats = np.mean(l), np.std(l), min(l), max(l), *list(np.quantile(l, [.5, .6, .7, .8, .9]))
    return stats

def triangles(G):
    l = list(nx.triangles(G).values()) # incident on nodes, not edges!
    stats = np.mean(l), np.std(l), min(l), max(l), *list(np.quantile(l, [.80, .90, .95, .97, .99]))
    return stats

def cores(G):
    l = list(nx.core_number(G).values())
    stats = np.mean(l), np.std(l), min(l), max(l), *list(np.quantile(l, [.25, .5, .75]))
    return stats

def closeness(G):
    l = list(nx.closeness_centrality(G).values())
    stats = np.mean(l), np.std(l), min(l), max(l), *list(np.quantile(l, [.25, .5, .75]))
    return stats

def pageranks(G):
    l = list(nx.pagerank(G, alpha=0.85).values())
    stats = np.mean(l), np.std(l), min(l), max(l), *list(np.quantile(l, [.25, .5, .75]))
    return stats

def add_to_csv(row):
    fp = os.path.join(os.getcwd(), "../", "Dataset","StaticFeatures.csv")
    with open(fp, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def get_ids():
    FP_RES = os.path.join(os.getcwd(), "../", "Dataset", "results.csv")
    df_res = pd.read_csv(FP_RES,header=0)
    df_res = df_res[df_res["Result"] != "Died out for 1000 tries"]
    names = list(df_res["Name"].values)

    # remove names already processed locally
    FP_IDS = os.path.join(os.getcwd(), "../", "Dataset", "StaticFeatures.csv")
    df = pd.read_csv(FP_IDS, header=0)
    processed = list(df["Name"].values)
    # final list to be done
    names = [n for n in names if n not in processed]

    # remove names already in folder
    names = [n for n in names if n+".json" not in os.listdir(os.getcwd() + "/rawresponse")]

    # only category 0
    # names = [n for n in names if n[0] == str(group)]
    return names

# %%
G = nx.read_graphml("justone.graphml")
row = list(degrees(G)) + list(clustering(G)) + list(triangles(G)) + [nx.transitivity(G), nx.degree_assortativity_coefficient(G)] + list(cores(G))
print(row)

# %%
if __name__ == '__main__':
    columns = ['avg deg', 'std deg', 'min deg', 'max deg', 'Q1 deg', 'Q2 deg', 'Q3 deg', 'Q4 deg', 'Q5 deg', 
               'avg clust', 'std clust', 'min clust', 'max clust', 'Q1 clust', 'Q2 clust', 'Q3 clust', 'Q4 clust', 'Q5 clust', 
               'avg tri', 'std tri', 'min tri', 'max tri', 'Q1 tri', 'Q2 tri', 'Q3 tri', 'Q4 tri', 'Q5 tri', 
               'transitivity',
               'assortativity',
               'avg core', 'std core', 'min core', 'max core', 'Q1 core', 'Q2 core', 'Q3 core'
              ]

    # check if static.csv exists, if not create it
    fp = os.path.join(os.getcwd(), "../", "Dataset","StaticFeatures.csv")
 
    if not os.path.exists(fp):
        with open(fp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name",] + columns)

    # final list to be done
    names = get_ids() 

    print(f"Calculating static features for still {len(names)} networks")

    # calculate features with parallel processes
    with Pool(processes=4) as pool:
        pool.map(calc_static_features, names)

# %%
# %%
