"""Main module."""

from .novelty_search import Archive, NoveltySearch
from .domain import Instance
import numpy as np

if __name__ == "__main__":
    variables = np.random.rand(10, 100)
    performances = np.random.rand(10, 4)
    features = np.random.rand(10, 8)
    instances = [Instance(variables=x.tolist()) for x in variables]
    for i in range(len(instances)):
        instances[i].features = features[i]
        instances[i].performance = performances[i]

    archive = Archive(instances)

    ns = NoveltySearch(t_a=0.85, t_ss=0.30, k=3, descriptor="features")
    print(f"NS algorithm initial status: {ns}")
    ns.sparseness(instances)
    ns.update_solution_set(instances)
    print(f"NS algorithm final status: {ns}")
