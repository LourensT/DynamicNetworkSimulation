# CTBP dynamic network simulation

This repository contains the code for generating dynamic networks. Dynamic networks are networks that grow one node at a time. The new node first samples its random out-degree, i.e. the number of existing nodes it will attach too. In this repository, this is sampled from the empirical out-degree distribution, which is stored as a python dictionary in `out_degree_distr.py`. The node then attaches to existing nodes, with probabilities defined by a given combinations of the following mechanisms:
* Affine preferential attachment
* Lognormal aging
* Fitness
    - uniform
    - exponential
    - powerlaw (pareto distributed)

Combinations of these mechanisms leads to $2 \times 2 \times 4 = 16$ different possible generative models. However, certain combinations have nice properties (e.g. scale-free) while others have not-so-nice properties (e.g. being explosive in size). 

In addition, this repository contains code for:
* calculating network statistics, referred to here as "static features"
* calculating *dynamic feature matrices*

## Simulation method
The simulation method builds a continuous time branching process. This is a tree. The tree is of size $\text{network size} \times \text{sum of out-degrees}$. The tree is then collapsed to form the  final network.


## Repository structure
The repository is structured as follows:

* `simulation.py` contains the main simulation.
    * it depends on `empirical_distribution_outdegree.py` for sampling the empirical outdegree of each vertex, which in turn uses `out_degree_distr_OC.pkl` as data.
    * `generate_dataset.py` contains the script that was used to generate the networks in the paper.

* `dynamic_feature_extraction.py` contains code for loading the dynamic network and calculating the dynamic feature matrices as specified in the paper.

* `static_feature_extraction.py` contains code for calculating the static features used in the paper. 
    * it depends on `dynamic_feature_extraction.py` to load the dynamic network and get the static final snapshot as a networkx Graph object.
    * some static featuers are computationaly heavy, and for large/many networks, one is recommended to use [graph-tool](https://graph-tool.skewed.de/), a Python module with an c++ implementation. See `calculating_static_featuers_with_graph_tool.py`.
