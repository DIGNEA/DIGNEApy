import argparse

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import tqdm

from digneapy.domains import TSPDomain
from digneapy.solvers import greedy, nneighbour, two_opt

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = [
    "Tahoma",
    "DejaVu Sans",
    "Lucida Grande",
    "Verdana",
]
plt.rcParams["font.size"] = 16


def main(number_of_nodes: int, amount: int):
    results = {}
    results.setdefault("two_opt", [])
    results.setdefault("nneighbour", [])
    results.setdefault("greedy", [])

    for _ in tqdm.tqdm(range(amount)):
        domain = TSPDomain(number_of_nodes=number_of_nodes)
        instance = domain.generate_instances(n=1)
        problem = domain.generate_problems_from_instances(instance)[0]
        two_fitness = two_opt(problem)[0].fitness
        nn_fitness = nneighbour(problem)[0].fitness
        greedy_fitness = greedy(problem)[0].fitness

        results["two_opt"].append(two_fitness)
        results["nneighbour"].append(nn_fitness)
        results["greedy"].append(greedy_fitness)

    df = pl.from_dict(results)
    blank = " " * 80

    print(f"\r{blank}\r", end="")
    print(df.head())
    df.write_csv("tsp_results.csv")

    plt.figure(figsize=(12, 8))
    axes = sns.boxplot(data=df)
    axes.set_ylabel("Profit")
    axes.set_xlabel("Solver")
    axes.get_figure().savefig("tsp_results.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tsp_heuristics",
        description="Python script to exemplify how to solve TSP instances using digneapy",
    )
    parser.add_argument(
        "number_of_nodes",
        choices=(10, 50, 100, 500, 1000),
        help="Dimension of the instances to solve",
        type=int,
    )
    parser.add_argument("amount", help="Number of instances to solve", type=int)
    args = parser.parse_args()
    main(args.number_of_nodes, args.amount)
