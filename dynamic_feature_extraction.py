import numpy as np
import os
import csv

from collections import OrderedDict
import networkx as nx

import matplotlib.pyplot as plt

class GraphAnalysis:

    '''
    class for analyzing dynamic network.
    - load network from nodes and edgesf iles
    - calculate dynamic feature matrix
    - plot degree distribution
    - plot network growth over time

    '''
    def __init__(self, id, fp = "\\output\\", verbose=False):
        # whether to log 
        self.verbose = verbose

        self.id = id
        self.fp = fp

        # graph representation
        self.edges = {} # dict edges {node : (set of adjacent nodes)}
        self.nodes = OrderedDict() # OrderedDictionary of form {node : time}

        filepath_edges = os.path.join(fp, str(id) + "_edges.csv")
        filepath_nodes = os.path.join(fp, str(id) +"_nodes.csv")

        with open(filepath_nodes, 'r') as f:
            self.load_nodes(f)

        with open(filepath_edges, 'r') as f:
            self.load_edges(f)

    '''
    get so-called scaled timesteps such that each timestep has the same number of nodes arriving
    '''
    def get_scaled_timesteps(self, a, up_to):
        number_of_nodes_per_step = int(up_to / a) # number of nodes arriving per timestep
        time_max = list(self.nodes.items())[up_to-1][1] # total running time
        stamps = [] # timestamps indicating arrival of last node for timestep
        for i in range(a):

            # get the timestamp of the arrival of the last node
            succes = False
            j = 0
            while not succes: # ceiling of i/number_of_nodes_per_step that is in nodes.
                try:
                    s = float(self.nodes[str((i*number_of_nodes_per_step)+j)])
                    stamps.append(s)
                    succes = True
                except KeyError:
                    j += 1

        stamps.append(time_max)
        return stamps

    '''
    calculates feature matrix from Cai2021

    @param a: number of rows; timeinterval
    @param b: number of column; 

    @param save: whether to save the matrix as csv
    @param up_to: cutoff node, i.e. if consider only part of the network evolution.

    @return a x b matrix of floats.
    '''
    def calculate_feature_matrix(self, a=3, b=4, fp="", save=False, name=None, up_to=None, scaled_timesteps=True, normalize=True):
        
        if up_to is None: 
            up_to = len(self.nodes)

        assert up_to <= len(self.nodes), "up_to parameter larger than number of nodes"

        matrix = np.zeros((a, b))

        if scaled_timesteps:
            time_bounds = self.get_scaled_timesteps(a, up_to)
        else: 
            time_max = list(self.nodes.items())[up_to-1][1] # total running time
            time_bounds = [i*(time_max / len(matrix)) for i in range(len(matrix)+1)]

        node_groups = self.set_node_groups(up_to, len(matrix[0]))
        # fill the matrix for each row
        for i in range(len(matrix)):
            matrix[i] = self.get_matrix_row(node_groups, time_bounds[i], time_bounds[i+1])
        
        if self.verbose:
            print("all matrix rows are calculated")

        if normalize:
            self.normalize_matrix(matrix)

        if save:
            if name is None: # use default name
                self.save_matrix_to_csv(matrix, node_groups, fp, "_matrix")
            else:
                self.save_matrix_to_csv(matrix, node_groups, fp, name)

        return matrix

    '''
    calculate added edges
    '''        
    def get_matrix_row(self, node_groups, t_start, t_end):
        new_nodes = []

        # find which nodes were added in timestep
        for n, t in self.nodes.items(): # TODO optimize this iteration by storing some sort of iterator
            t = float(t)
            if t >= t_start:
                if t <= t_end:
                    new_nodes.append(n)
                else:
                    break

        row = np.zeros(len(node_groups))

        # update matrix
        for n in new_nodes:
            for v in self.edges[n]:
                for i, group in node_groups.items():
                    if v in group:
                        row[i] += 1

        # average out values, the matrix represents the average number of degrees a cell in the node group has received.
        for i, group in node_groups.items():
            if len(group) != 0:
                row[i] = row[i] / len(group)
            else:
                if self.verbose:
                    print(" Group size of 0! ")

        return row
    
    '''
    normalize matrix by dividing every element in matrix by sum over all.
    '''
    def normalize_matrix(self, matrix):
        total = sum(sum(matrix[i]) for i in range(len(matrix)))

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = matrix[i][j] / total
        
        if self.verbose:
            print("matrix has been normalized")

    '''
    set the node_groups to self.node_groups (G_j)
    '''
    def set_node_groups(self, up_to, group_amount):
        degree_sequence = {}

        # calculate the degree sequence untill the "up_to"
        for n in list(self.nodes.keys())[:up_to]:
            if n in self.edges:
                degree_sequence[n] = len(self.edges[n])
                for v in self.edges[n]: # since OrderedDict and PA only connects backwards, all v's already in degree_sequence
                    if v in degree_sequence:
                        degree_sequence[v] += 1     
                    else:
                        degree_sequence[v] = 1
            else:
                self.edges[n] = set()
                degree_sequence[n] = 0
        
        # get a sorted degree-sequence
        degrees = list(degree_sequence.values())
        degrees.sort()

        # calculate the empirical CDF (F_T in Weiting Cai)
        empirical_cdf = {}
        n = len(degrees)

        prev = 0
        empirical_cdf[0] = 0
        for v in degrees:
            if v > prev:
                empirical_cdf[v] = empirical_cdf[prev] + 1/n
                prev = v
            else:
                empirical_cdf[prev] += 1/n

        assert abs(empirical_cdf[degrees[-1]] - 1) < 0.00001
        empirical_cdf[degrees[-1]] = 0.999999999 # last cannot be >= 1 by overflow

        node_groups = {i : set() for i in range(group_amount)}
        for node, degree in degree_sequence.items():
            i = group_amount - 1 - ((empirical_cdf[degree]) // (1/group_amount))
            node_groups[i].add(node)

        if self.verbose:
            print("node_groups are set")

        return node_groups

    '''
    populate self.nodes from file
    '''
    def load_nodes(self, file, up_to=None):
        if not up_to is None:
            assert isinstance(up_to, int) and up_to > 0, "up_to should be positive integer"

        reader = csv.reader(file)
        
        counter = 0
        for node in reader:         # node is stored as name_node : timestamp
            self.nodes[node[0]] = float(node[1])
            counter += 1

        if self.verbose:
            print(f"loaded all {counter} nodes")

    '''
    populate self.edges from file.  
    '''
    def load_edges(self, file):
        reader = csv.reader(file)

        # we add the root manually as it has no originating edges
        self.edges['0'] = set()

        counter = 0
        for edge in reader:         # edge in file is stored as node1, node2, timestamp

            if edge[0] in self.edges:
                self.edges[edge[0]].add(edge[1])
            else:
                self.edges[edge[0]] = set((edge[1],))
            
            if self.verbose:
                counter += 1
                if counter % 1_000_000 == 0: # print update every million edges
                    print(f"++ loaded {counter} edges ")

        if self.verbose:
            print(f"Loaded edges")

    def save_matrix_to_csv(self, matrix, node_groups, fp, post_script):
        with open(os.path.join(fp, self.id + post_script + ".csv"), "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([len(group) for group in node_groups.values()])

            for row in matrix:
                writer.writerow(row)
    
    '''
    Set self.G to the networkx.Graph() object representation
    '''
    def set_graph_representation(self):
        self.G = nx.Graph()

        for node in self.nodes.keys():
            self.G.add_node(node)
        
        for u in self.edges.keys():
            for v in self.edges[u]:
                self.G.add_edge(u, v)

        # remove self loops  
        if self.G.has_edge('0', '0'):
            self.G.remove_edge('0', '0')
    
    '''
    gets the networkx.Graph object representation, if it does not exist, creates it.
    '''
    def get_graph_representation(self):
        if hasattr(self, 'G'):
            return self.G
        else:
            self.set_graph_representation()
            return self.G
    '''
    plot degree distribution
    '''
    def plot_tail_degree_distribution(self, loglog=True, title=None, save=False, show=True):
        assert (not save) or not title is None, "need title to save"
        assert save or show, "neither saving nor showing, what u want?"

        # https://github.com/LourensT/degreedistributions
        from DegreeDistributions.DegreeDistributions import DegreeDistribution
        distr = DegreeDistribution(self.get_graph_representation() ,tail=True)
        plt.scatter(x=distr.keys(), y=distr.values(), color='red')

        if loglog:
            plt.yscale("log")
            plt.xscale("log")

        if not title is None:
            plt.title(title)
        
        if save:
            plt.savefig(self.fp + title+"_degrees.png")
            print(f"Saved {title}_degrees.png")

        if show:
            plt.show()
        else:
            plt.clf()
    
    '''
    plot the size over time. @pre's are checked, so method is reasonably robust.

    optional params:
    @title: string - title of the plot
    @save: bool    - if True, save as @title.png
            @pre: title != None
    @fit_exp: bool - if True, an exponential curve, OLS fit of y = a*exp(bx)

    '''
    def plot_size_over_time(self, title=None, save=False, fit_exp = False, show = True, log=False):
        assert (not save) or not title is None, "need title to save"
        assert save or show, "neither saving nor showing, what u want?"

        cumulative_growth = {}
        prev = 0
        cumulative_growth[prev] = 0
        for time in self.nodes.values():
            if time in cumulative_growth:
                cumulative_growth[time] += 1
            else:
                assert prev <= time, "not ordered; what the fuck"
                cumulative_growth[time] = 1 + cumulative_growth[prev]
                prev = time

        plt.scatter(x=cumulative_growth.keys(), y=cumulative_growth.values())

        legend = ["Population size over time"]

        # fit the exponential curve y = a*exp(bx)
        if fit_exp:
            logY = [np.log(i) for i in cumulative_growth.values() if i > 0][10:]
            x = [ [i,] for i in cumulative_growth.keys()][10:] 
            
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(x, logY)
            a = np.exp(reg.intercept_)
            b = reg.coef_[0]

            x_range = np.linspace(x[9][0], max(cumulative_growth.keys()), num=100)
            y_pred = [a*np.exp(b*n) for n in x_range]

            plt.plot(x_range, y_pred, color="red")
            legend.append(f"fit of y = a*exp(bx) with a = {a}, b = {b}")

        if not title is None:
            plt.title(title)

        if log:
            plt.yscale("log")

        if save:
            plt.savefig(self.fp + title+"_growth.png")
            print(f"Saved {title}_growth.png")

        if show:
            plt.show()
        else:
            plt.clf()


