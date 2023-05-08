from dynamic_feature_extraction import *
from simulation import *
from empirical_distribution_outdegree import sample_empirical_outdegree

from typing import List
import os
import random

from multiprocessing import Pool

import scipy.stats
from datetime import datetime

FILEPATH = os.path.join(os.getcwd(), "Dataset")

# absolute filepath in which to put the features
FEATUREMATRIX_FP = os.path.join(FILEPATH, "DatasetFeatures")
# if folder does not exist, create it
if not os.path.isdir(FEATUREMATRIX_FP):
    os.mkdir(FEATUREMATRIX_FP)
# STORE NETWORKS
SOURCE_FP = os.path.join(FILEPATH, "DatasetNetworkSource")
# if folder does not exist, create it
if not os.path.isdir(SOURCE_FP):
    os.mkdir(SOURCE_FP)

# number of networks per category
START_NUMBER = 0
FINAL_NUMBER = 750
NR_OF_PROCESSES = 8

# max size of network being analyzed
NETWORK_SIZE = 20000
# number of different sizes to calculate features for

# maximum width or height for feature matrix, minumum 2
MATRIX_DIMENSIONS = (10,10)

'''
Dumps all feature files

@arg name: string, which is used to load the correct files and name the output directory
@arg fp: string, filepath where the name_nodes.csv and name_edges.csv can be found
@arg destination: where to dump the feature folder
@arg sizes. ...
@arg dimension, ...
'''
def dump_features(name : str):
    g = GraphAnalysis(name, SOURCE_FP, verbose=False)
    g.calculate_feature_matrix(a=MATRIX_DIMENSIONS[0], b=MATRIX_DIMENSIONS[1], fp=FEATUREMATRIX_FP, scaled_timesteps=True, save=True)

def append_result(name, params, result):
    row = [name,] + params + [result]
    assert len(row) == 7, "incorrect length of entry"

    filename = os.path.join(FILEPATH, "results.csv")
    if os.path.isfile(filename):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    else:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Code", "Network Size", "PA", "Fitness", "Aging", "Result"])
            writer.writerow(row)

def name_exists(name):
    return os.path.isfile(os.path.join(FEATUREMATRIX_FP, name + "_matrix.csv"))

def generate_network(name, code, *args):
    if name_exists(name):
        print(datetime.now().strftime("%H:%M:%S"), f"{name} already exists")
        return

    print(datetime.now().strftime("%H:%M:%S"), f"generating {name}")
    info = [code, NETWORK_SIZE] + list(args)

    res = False
    count = 0
    while not res:
        ctbp = simulator_from_parameters(NETWORK_SIZE, sample_empirical_outdegree, *args)
        res = ctbp.generate(save_to=[SOURCE_FP, name], debug=False)

        count += 1
        if count > 1000: # died out too many times
            append_result(name, info, "Died out for 1000 tries")
            return

    append_result(name, info, count) 

    dump_features(name)

# 0 - Uniform Attachment Model
# 1 - Affine PA(a,b)
# 2 - Power-law (xmin, tau) Fitness
# 3 - Exponential (lambda) Fitness 
# 4 - Power-law Fitness (xmin, tau), 
#     Aging (mu, st.dev).
# 5 - Exponential Fitness (lambda),
#     Aging (mu, st.dev)
# 6 - PA(a,b) with Exponential (lambda) Fitness
# 7 - PA(a,b),
#     Aging(mu, st.dev).
# 8 - PA(a,b), 
#     Exponential fitness (lambda),  
#     Aging (mu, st.dev).

def main_procedure():
    print(datetime.now().strftime("%H:%M:%S"), "Generating parameters")

    # # 0 - Uniform Attachment
    params = []
    for i in range(START_NUMBER, FINAL_NUMBER):
        name = "0-"+(str(i))
        params.append([name, 0, None, None, None])

    # # 1 - Affine PA(a,b)
    for i in range(START_NUMBER, FINAL_NUMBER):
        name = "1-"+(str(i))
        pa = (scipy.stats.uniform(loc=1, scale=3).rvs(), scipy.stats.uniform(loc=1, scale=3).rvs())
        params.append([name, 1,  pa, None, None])

    # # 2 - Power-law (xmin, tau) Fitness
    for i in range(START_NUMBER, FINAL_NUMBER):
        name = "2-"+(str(i))
        fitness = (scipy.stats.uniform(0, 1).rvs(), scipy.stats.uniform(loc=2, scale=2).rvs())
        params.append([name, 2,  None, fitness, None])

    # # 3 - Exponential (lambda) Fitness 
    for i in range(START_NUMBER, FINAL_NUMBER):
        name = "3-"+(str(i))
        fitness = scipy.stats.uniform(loc=0.1, scale=2.9).rvs()
        params.append([name, 3,  None, fitness, None])

    # 4 - Power-law Fitness (xmin, tau), 
    #     Aging (mu, st.dev).
    for i in range(START_NUMBER, FINAL_NUMBER):
        name = "4-"+(str(i))
        fitness = (scipy.stats.uniform(loc=0.5, scale=0.5).rvs(), scipy.stats.uniform(loc=2, scale=0.7).rvs())
        aging = (scipy.stats.uniform(loc=0.1, scale=2.9).rvs(), 1)
        params.append([name, 4,  None, fitness, aging])

    # 5 - Exponential Fitness (lambda),
    #     Aging (mu, st.dev)
    for i in range(START_NUMBER, FINAL_NUMBER):
        name = "5-"+(str(i))
        fitness = scipy.stats.uniform(loc=0.1, scale=0.9).rvs()
        aging = (scipy.stats.uniform(loc=0.1, scale=2.9).rvs(), 1)
        params.append([name, 5,  None, fitness, aging])

    # 6 - PA(a,b) with Uniform (a, b) Fitness
    for i in range(START_NUMBER, FINAL_NUMBER):
        name = "6-"+(str(i))
        pa = (scipy.stats.uniform(loc=1, scale=3).rvs(), scipy.stats.uniform(loc=1, scale=3).rvs())
        fitness = ("uniform", scipy.stats.uniform(loc=0.1, scale=0.9).rvs(), scipy.stats.uniform(loc=1, scale=4).rvs())
        params.append([name, 6,  pa, fitness, None])    

    # 7 - PA(a,b),
    #     Aging(mu, st.dev).
    for i in range(START_NUMBER, FINAL_NUMBER):
        name = "7-"+(str(i))
        pa = (scipy.stats.uniform(loc=3.3, scale=3.7).rvs(), scipy.stats.uniform(loc=1, scale=3).rvs())
        aging = (scipy.stats.uniform(loc=0.1, scale=29).rvs(), 1)
        params.append([name, 7,  pa, None, aging])

    # 8 - PA(a,b), 
    #     Exponential fitness (lambda),  
    #     Aging (mu, st.dev).
    for i in range(START_NUMBER, FINAL_NUMBER):
        name = "8-"+(str(i))
        pa = (scipy.stats.uniform(loc=1, scale=3).rvs(), scipy.stats.uniform(loc=1, scale=3).rvs())
        fitness = scipy.stats.uniform(loc=0.1, scale=((pa[0] + (pa[1]/10.29))- 0.1 )).rvs() #lambda  ∈ [0.1,  a + b/E[M]]
        aging = (scipy.stats.uniform(loc=0.1, scale=2.9).rvs(), 1)
        params.append([name, 8, pa, fitness, aging])

    random.shuffle(params)

    print(datetime.now().strftime("%H:%M:%S"), f"Generated Params, moving on to generating {len(params)} networks.")

    with Pool(processes=NR_OF_PROCESSES) as pool:
        pool.starmap(generate_network, params)

if __name__ == "__main__":
    main_procedure()
