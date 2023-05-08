# %%
import pickle
import random
import os
with open(os.path.join(os.getcwd(), "out_degree_distr_OC.pkl"), "rb") as f:
    out_degree_distr_OC = pickle.load(f)
    options = list(out_degree_distr_OC.keys())
    weights = list(out_degree_distr_OC.values())

def sample_outdegree(n):
    return n*[20]

def sample_empirical_outdegree(n):
    return random.choices(options, weights=weights, k=n)

if __name__=="__main__":
    values = sample_empirical_outdegree(1000)
    # plot histogram
    import matplotlib.pyplot as plt
    plt.hist(values, bins=100, density=True);

# %%
