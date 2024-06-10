# DIGNEApy
---
Diverse Instance Generator with Novelty Search and Evolutionary Algorithms
  
[![Test](https://github.com/DIGNEA/DIGNEApy/actions/workflows/python-app.yml/badge.svg)](https://github.com/DIGNEA/DIGNEApy/actions/workflows/python-app.yml)
[![Coverage Status](https://coveralls.io/repos/github/DIGNEA/DIGNEApy/badge.svg?branch=main)](https://coveralls.io/github/DIGNEA/DIGNEApy?branch=main)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


Repository containing the Python version of DIGNEA, a Diverse Instance Generator with Novelty Search and Evolutionary Algorithms. This framework is an extensible tool for generating diverse and discriminatory instances for any desired domain. The instances obtained generated will be biased to the performance of a *target* in a specified portfolio of algorithms. 




### Dependencies

- Numpy
- Sklearn
- Pandas
- Keras
- DEAP
- Tensorflow 
- PyTorch
- Pybind11
- Seaborn
- Matplotlib
    

## Publications

DIGNEA was used in the following publications:

* Alejandro Marrero, Eduardo Segredo, and Coromoto Leon. 2021. A parallel genetic algorithm to speed up the resolution of the algorithm selection problem. Proceedings of the Genetic and Evolutionary Computation Conference Companion. Association for Computing Machinery, New York, NY, USA, 1978–1981. DOI:https://doi.org/10.1145/3449726.3463160

* Marrero, A., Segredo, E., León, C., Hart, E. 2022. A Novelty-Search Approach to Filling an Instance-Space with Diverse and Discriminatory Instances for the Knapsack Problem. In: Rudolph, G., Kononova, A.V., Aguirre, H., Kerschke, P., Ochoa, G., Tušar, T. (eds) Parallel Problem Solving from Nature – PPSN XVII. PPSN 2022. Lecture Notes in Computer Science, vol 13398. Springer, Cham. https://doi.org/10.1007/978-3-031-14714-2_16

* Alejandro Marrero, Eduardo Segredo, Emma Hart, Jakob Bossek, and Aneta Neumann. 2023. Generating diverse and discriminatory knapsack instances by searching for novelty in variable dimensions of feature-space. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '23). Association for Computing Machinery, New York, NY, USA, 312–320. https://doi.org/10.1145/3583131.3590504
  
* Marrero, A., Segredo, E., León, C., & Hart, E. 2024. Learning Descriptors for Novelty-Search Based Instance Generation via Meta-evolution. In Genetic and Evolutionary Computation Conference (GECCO ’24), July 14–18, 2024, Melbourne, VIC, Australia. https://doi.org/10.1145/3638529.3654028

* Alejandro Marrero, Eduardo Segredo, Coromoto León, Emma Hart; Synthesising Diverse and Discriminatory Sets of Instances using Novelty Search in Combinatorial Domains. Evolutionary Computation 2024; doi: https://doi.org/10.1162/evco_a_00350

* Marrero, A. 2024. Evolutionary Computation Methods for Instance Generation in Optimisation Domains. PhD thesis. Universidad de La Laguna. https://riull.ull.es/xmlui/handle/915/37726

## How to cite DIGNEA

If you use DIGNEA in your research work, remember to cite: 
>
>@article{dignea_23,
>title = {DIGNEA: A tool to generate diverse and discriminatory instance suites for optimisation domains},
>journal = {SoftwareX},
>volume = {22},
>pages = {101355},
>year = {2023},
>issn = {2352-7110},
>doi = {https://doi.org/10.1016/j.softx.2023.101355},
>url = {https://www.sciencedirect.com/science/article/pii/S2352711023000511},
>author = {Alejandro Marrero and Eduardo Segredo and Coromoto León and Emma Hart},
>keywords = {Instance generation, Novelty search, Evolutionary algorithm, Optimisation, Knapsack problem},
>abstract = {To advance research in the development of optimisation algorithms, it is crucial to have access to large test-beds of diverse and discriminatory instances from a domain that can highlight strengths and weaknesses of different algorithms. The DIGNEA tool enables diverse instance suites to be generated for any domain, that are also discriminatory with respect to a set of solvers of the user choice. Written in C++, and delivered as a repository and as a Docker image, its modular and template-based design enables it to be easily adapted to multiple domains and types of solvers with minimal effort. This paper exemplifies how to generate instances for the Knapsack Problem domain.}
>}
>