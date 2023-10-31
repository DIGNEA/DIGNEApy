"""Main module."""

from .novelty_search import Archive, NoveltySearch
from .domains.knapsack import KPDomain, Knapsack
from .solvers.heuristics import DefaultKP, MaP, MiW
from .generator import EIG
import numpy as np

if __name__ == "__main__":
    # variables = np.random.rand(10, 100)
    # performances = np.random.rand(10, 4)
    # features = np.random.rand(10, 8)
    # instances = [Instance(variables=x.tolist()) for x in variables]
    # for i in range(len(instances)):
    #     instances[i].features = features[i]
    #     instances[i].performance = performances[i]

    # archive = Archive(instances)

    # ns = NoveltySearch(t_a=0.5, t_ss=0.10, k=3, descriptor="features")
    # print(f"NS algorithm initial status: {ns}")
    # ns.sparseness(instances, verbose=True)
    # ns.update_solution_set(instances, verbose=True)
    # print(f"NS algorithm final status: {ns}")

    kp_domain = KPDomain(dimension=100)
    # instance = kp_domain.generate_instance()
    # knapsack = KPDomain.from_instance(instance)
    # print(f"Instance: {instance!r}")
    default = DefaultKP()
    map_heuristic = MaP()
    miw = MiW()
    # print(f"{knapsack!r}")
    # fitness, chromosome = default.run(problem=knapsack)
    # print(f"fitness: {fitness!r}, chromosome: {chromosome!r}")
    # fitness, chromosome = map_heuristic.run(problem=knapsack)
    # print(f"fitness: {fitness!r}, chromosome: {chromosome!r}")
    # fitness, chromosome = miw.run(problem=knapsack)
    # print(f"fitness: {fitness!r}, chromosome: {chromosome!r}")
    eig = EIG(
        100,
        1000,
        domain=kp_domain,
        portfolio=[default, map_heuristic, miw],
        t_a=1000,
        t_ss=500,
    )
    print(f"{eig!r}")
    print(eig)
    eig.run()
