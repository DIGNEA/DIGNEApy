
# Create an algorithm 

[TOC]


To exemplify how to create a new algorithm in DIGNEA, we will see how to include the classical [Nearest Neighbour](https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm) algorithm for TSP. This can be simply defined as a function:

## NN as a function.
The function receives an object of the TSP with the required information to evaluate the solutions it generates:

```python
def nneighbour(problem: TSP) -> list[Solution]:
    """Nearest-Neighbour Heuristic for the Travelling Salesman Problem

    Args:
        problem (TSP): Problem to solve

    Raises:
        ValueError: If problem is None

    Returns:
        list[Solution]: Collection of solutions to the problem.
    """
    if problem is None:
        raise ValueError("No problem found in nneighbour heuristic")

    distances = problem._distances
    current_node = 0
    visited_nodes: set[int] = set([current_node])
    tour = np.zeros(problem.dimension + 1)
    length = np.float32(0)
    idx = 1
    while len(visited_nodes) != problem.dimension:
        next_node = 0
        min_distance = np.finfo(np.float32).max

        for j in range(problem.dimension):
            if j not in visited_nodes and distances[current_node][j] < min_distance:
                min_distance = distances[current_node][j]
                next_node = j

        visited_nodes.add(next_node)
        tour[idx] = next_node
        idx += 1
        length += min_distance
        current_node = next_node

    length += distances[current_node][0]
    length = 1.0 / length
    return [Solution(chromosome=tour, objectives=(length,), fitness=length)]
```

## NN as as class

Alternatively, you could defined your solver as a Python class like:

```python
class NearestNeighbour(Solver, SupportsSolve[P]):
    def __init__(self): ...

    def __call__(self, problem: TSP) -> list[Solution]:
        if problem is None:
            raise ValueError("No problem found in nneighbour heuristic")

        distances = problem._distances
        current_node = 0
        visited_nodes: set[int] = set([current_node])
        tour = np.zeros(problem.dimension + 1)
        length = np.float32(0)
        idx = 1
        while len(visited_nodes) != problem.dimension:
            next_node = 0
            min_distance = np.finfo(np.float32).max

            for j in range(problem.dimension):
                if j not in visited_nodes and distances[current_node][j] < min_distance:
                    min_distance = distances[current_node][j]
                    next_node = j

            visited_nodes.add(next_node)
            tour[idx] = next_node
            idx += 1
            length += min_distance
            current_node = next_node

        length += distances[current_node][0]
        length = 1.0 / length
        return [Solution(chromosome=tour, objectives=(length,), fitness=length)]
```

## How to

1. [Run an experiment](01_eig_example.md)
2. [Create a domain](02_create_domain.md)
3. [Create a solver](03_create_algorithm.md)