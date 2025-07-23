# Instance Generation Example

[TOC]



# Generating Knapsack Problem Instances with DIGNEApy

Here's an example of how to generated diverse and discriminatory instances for the [Knapsack Problem](https://en.wikipedia.org/wiki/Knapsack_problem). 
We will be creating a set of KP instances which will be adapted to the performance of a specific heuristic. 
For this experiment we will:

1. Define EAGenerator's parameter configuration.
2. Set the dimension of the KP instances to generate.
3. Specify the portfolio of algorithms to solve the instances. In this case, four deterministic heuristics.
4. Run the experiment.
5. Finally, collect the results in a Pandas DataFrame.

```python
    domain = KnapsackDomain(50, capacity_approach="percentage")
    eig = EAGenerator(
        pop_size=128,
        generations=100,
        domain=domain,
        portfolio=[default_kp, map_kp, miw_kp, mpw_kp],
        novelty_approach=NS(Archive(threshold=3.0), k=15),
        solution_set=Archive(threshold=3.0),
        repetitions=1,
        descriptor_strategy='features',
        replacement=generational_replacement,
    )

    result = eig()
    df = pd.DataFrame(list(i.to_series() for i in result.instances))
    df.insert(0, "target", result.target)
```



## How to

1. [Run an experiment](01_eig_example.md)
2. [Create a domain](02_create_domain.md)
3. [Create a solver](03_create_algorithm.md)