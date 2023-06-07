from dynamic_feature_extraction import *

from typing import List
import os

from multiprocessing import Pool

from datetime import datetime

FILEPATH = os.path.join(os.getcwd(), "../", "Dataset")

# absolute filepath in which to put the features
ALTFEATUREMATRIX_FP = os.path.join(FILEPATH, "AlternativeDynamicFeatures")
# if folder does not exist, create it
if not os.path.isdir(ALTFEATUREMATRIX_FP):
    os.mkdir(ALTFEATUREMATRIX_FP)
# STORE NETWORKS
SOURCE_FP = os.path.join(FILEPATH, "DatasetNetworkSource")

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
    g.calculate_feature_matrix(a=MATRIX_DIMENSIONS[0], b=MATRIX_DIMENSIONS[1], fp=ALTFEATUREMATRIX_FP, scaled_timesteps=False, save=True)

def main_procedure():
    files = os.listdir(SOURCE_FP)
    files = [f.replace("_edges.csv", "") for f in files if f.endswith("_edges.csv")]
    done = os.listdir(ALTFEATUREMATRIX_FP)
    done = [f.replace("matrix.csv", "") for f in done]
    print("number of networks left to process: ", len(files) - len(done))
    for f in files:
        if f not in done:
            dump_features(f)

if __name__=="__main__":
    main_procedure()