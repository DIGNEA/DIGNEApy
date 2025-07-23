
# Create a new domain

[TOC]


This tutorial will show how to include a new domain in DIGNEApy. For this purpose, we will create the [Travelling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem) domain. It is important to remark that the addition of a new domain requires the definition of the Problem and Domain subclasses for such specific domain.

# Creating the Travelling Salesman Problem (TSP)

Let's start by creating a Travelling Salesman Problem class which will allow us to solve TSP instances. The TSP class must be a subclass of the Problem class and define (at least) the ```__call__```, ```evaluate```, ```create_solution``` and ```to_instance``` methods. For this example, we should considering some domain-dependent attributes such as the number of nodes and the coordinates of each node in the instance.

```python
class TSP(Problem):
    """Symmetric Travelling Salesman Problem"""

    def __init__(
        self,
        nodes: int,
        coords: Tuple[Tuple[int, int], ...],
        seed: int = 42,
        *args,
        **kwargs,
    ):
        """Creates a new Symmetric Travelling Salesman Problem

        Args:
            nodes (int): Number of nodes/cities in the instance to solve
            coords (Tuple[Tuple[int, int], ...]): Coordinates of each node/city.
        """
        self._nodes = nodes
        self._coords = np.array(coords)
        x_min, y_min = np.min(self._coords, axis=0)
        x_max, y_max = np.max(self._coords, axis=0)
        bounds = list(((x_min, y_min), (x_max, y_max)) for _ in range(self._nodes))
        super().__init__(dimension=nodes, bounds=bounds, name="TSP", seed=seed)

        self._distances = np.zeros((self._nodes, self._nodes))
        for i in range(self._nodes):
            for j in range(i + 1, self._nodes):
                self._distances[i][j] = np.linalg.norm(
                    self._coords[i] - self._coords[j]
                )
                self._distances[j][i] = self._distances[i][j]
```

Then, we could implement the evaluation method. This can be written directly in the ```__call__``` method or alternatively you could split the evaluation in two methods like:

```python
def __evaluate_constraints(self, individual: Sequence | Solution) -> bool:
        counter = Counter(individual)
        if any(counter[c] != 1 for c in counter if c != 0) or (
            individual[0] != 0 or individual[-1] != 0
        ):
            return False
        return True
```
and 

```python
def evaluate(self, individual: Sequence | Solution) -> tuple[float]:
        """Evaluates the candidate individual with the information of the Travelling Salesmas Problem.

        The fitness of the solution is the fraction of the sum of the distances of the tour
        Args:
            individual (Sequence | Solution): Individual to evaluate

        Returns:
            Tuple[float]: Fitness
        """
        if len(individual) != self._nodes + 1:
            msg = f"Mismatch between individual variables ({len(individual)}) and instance variables ({self._nodes}) in {self.__class__.__name__}. A solution for the TSP must be a sequence of len {self._nodes + 1}"
            raise ValueError(msg)

        penalty: np.float64 = np.float64(0)

        if self.__evaluate_constraints(individual):
            distance: float = 0.0
            for i in range(len(individual) - 2):
                distance += self._distances[individual[i]][individual[i + 1]]

            fitness = 1.0 / distance
        else:
            fitness = 2.938736e-39  # --> 1.0 / np.float.max
            penalty = np.finfo(np.float64).max

        if isinstance(individual, Solution):
            individual.fitness = fitness
            individual.objectives = (fitness,)
            individual.constraints = (penalty,)

        return (fitness,)

def __call__(self, individual: Sequence | Solution) -> tuple[float]:
        return self.evaluate(individual)
```

We must implement the method ```create_solution``` so we can initialise the algorithms solutions.

```python
def create_solution(self) -> Solution:
        items = [0] + list(range(1, self._nodes)) + [0]
        return Solution(chromosome=items)
```

To cast the TSP problem to an actual evolvable instance we must define the ```to_instance``` method. In this example, the variables/chromosome of the Instance is the coordinates of the nodes. Note that for each domain, the chromosome of the instances must be adapted to the relevant information.
```python
def to_instance(self) -> Instance:
        return Instance(variables=self._coords.flatten())
```

Finally, the whole TSP class with several added methods looks like this:


```python
class TSP(Problem):
    """Symmetric Travelling Salesman Problem"""

    def __init__(
        self,
        nodes: int,
        coords: Tuple[Tuple[int, int], ...],
        seed: int = 42,
        *args,
        **kwargs,
    ):
        """Creates a new Symmetric Travelling Salesman Problem

        Args:
            nodes (int): Number of nodes/cities in the instance to solve
            coords (Tuple[Tuple[int, int], ...]): Coordinates of each node/city.
        """
        self._nodes = nodes
        self._coords = np.array(coords)
        x_min, y_min = np.min(self._coords, axis=0)
        x_max, y_max = np.max(self._coords, axis=0)
        bounds = list(((x_min, y_min), (x_max, y_max)) for _ in range(self._nodes))
        super().__init__(dimension=nodes, bounds=bounds, name="TSP", seed=seed)

        self._distances = np.zeros((self._nodes, self._nodes))
        for i in range(self._nodes):
            for j in range(i + 1, self._nodes):
                self._distances[i][j] = np.linalg.norm(
                    self._coords[i] - self._coords[j]
                )
                self._distances[j][i] = self._distances[i][j]

    def __evaluate_constraints(self, individual: Sequence | Solution) -> bool:
        counter = Counter(individual)
        if any(counter[c] != 1 for c in counter if c != 0) or (
            individual[0] != 0 or individual[-1] != 0
        ):
            return False
        return True

    def evaluate(self, individual: Sequence | Solution) -> tuple[float]:
        """Evaluates the candidate individual with the information of the Travelling Salesmas Problem.

        The fitness of the solution is the fraction of the sum of the distances of the tour
        Args:
            individual (Sequence | Solution): Individual to evaluate

        Returns:
            Tuple[float]: Fitness
        """
        if len(individual) != self._nodes + 1:
            msg = f"Mismatch between individual variables ({len(individual)}) and instance variables ({self._nodes}) in {self.__class__.__name__}. A solution for the TSP must be a sequence of len {self._nodes + 1}"
            raise ValueError(msg)

        penalty: np.float64 = np.float64(0)

        if self.__evaluate_constraints(individual):
            distance: float = 0.0
            for i in range(len(individual) - 2):
                distance += self._distances[individual[i]][individual[i + 1]]

            fitness = 1.0 / distance
        else:
            fitness = 2.938736e-39  # --> 1.0 / np.float.max
            penalty = np.finfo(np.float64).max

        if isinstance(individual, Solution):
            individual.fitness = fitness
            individual.objectives = (fitness,)
            individual.constraints = (penalty,)

        return (fitness,)

    def __call__(self, individual: Sequence | Solution) -> tuple[float]:
        return self.evaluate(individual)

    def __repr__(self):
        return f"TSP<n={self._nodes}>"

    def __len__(self):
        return self._nodes

    def create_solution(self) -> Solution:
        items = [0] + list(range(1, self._nodes)) + [0]
        return Solution(chromosome=items)

    def to_file(self, filename: str = "instance.tsp"):
        with open(filename, "w") as file:
            file.write(f"{len(self)}\n\n")
            content = "\n".join(f"{x}\t{y}" for (x, y) in self._coords)
            file.write(content)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        nodes = int(lines[0])
        coords = []
        for line in lines[2:]:
            x, y = line.split()
            coords.append((int(x), int(y)))

        return cls(nodes=nodes, coords=tuple(coords))

    def to_instance(self) -> Instance:
        return Instance(variables=self._coords.flatten())
```

# Creating the Travelling Salesman Problem Domain (TSPDomain)

Once we have created the optimisation problem, we can define the domain of such problem. In this example, we call it TSPDomain. Every domain must define (at least) the ```generate_instance```, ```extract_features```, ```extract_features_as_dict``` and ```from_instance``` methods. The features methods are only relevant if you are planning to use the generators with the features-based descriptors. Otherwise you can return NotImplemented. 

A domain should include all the attributes and relevant information to generate the instances with particular characteristics. In this example, we need the dimension of the instances (number of nodes) and the ranges for the coordinates (xmin, xmax) and (ymin, ymax) for each node in the instance.


```python
class TSPDomain(Domain):
    """Domain to generate instances for the Symmetric Travelling Salesman Problem."""

    def __init__(
        self,
        dimension: int = 100,
        x_range: Tuple[int, int] = (0, 1000),
        y_range: Tuple[int, int] = (0, 1000),
        seed: int = 42,
    ):
        """Creates a new TSPDomain to generate instances for the Symmetric Travelling Salesman Problem

        Args:
            dimension (int, optional): Dimension of the instances to generate. Defaults to 100.
            x_range (Tuple[int, int], optional): Ranges for the Xs coordinates of each node/city. Defaults to (0, 1000).
            y_range (Tuple[int, int], optional): Ranges for the ys coordinates of each node/city. Defaults to (0, 1000).

        Raises:
            ValueError: If dimension is < 0
            ValueError: If x_range OR y_range does not have 2 dimensions each
            ValueError: If minimum ranges are greater than maximum ranges
        """
        if dimension < 0:
            raise ValueError(f"Expected dimension > 0 got {dimension}")
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError(
                f"Expected x_range and y_range to be a tuple with only to integers. Got: x_range = {x_range} and y_range = {y_range}"
            )
        x_min, x_max = x_range
        y_min, y_max = y_range
        if x_min < 0 or x_max <= x_min:
            raise ValueError(
                f"Expected x_range to be (x_min, x_max) where x_min >= 0 and x_max > x_min. Got: x_range {x_range}"
            )
        if y_min < 0 or y_max <= y_min:
            raise ValueError(
                f"Expected y_range to be (y_min, y_max) where y_min >= 0 and y_max > y_min. Got: y_range {y_range}"
            )

        self._x_range = x_range
        self._y_range = y_range
        __bounds = [
            (x_min, x_max) if i % 2 == 0 else (y_min, y_max)
            for i in range(dimension * 2)
        ]

        super().__init__(dimension=dimension, bounds=__bounds, name="TSP", seed=seed)
```

The ```generate_instance``` method is quite similar to the ```create_solution``` in the Problem class. Basically, it creates a random instance with the characteristics defined in the domain.

```python
def generate_instance(self) -> Instance:
        """Generates a new instances for the TSP domain

        Returns:
            Instance: New randomly generated instance
        """
        coords = self._rng.integers(
            low=(self._x_range[0], self._y_range[0]),
            high=(self._x_range[1], self._y_range[1]),
            size=(self.dimension, 2),
            dtype=int,
        )
        coords = coords.flatten()
        return Instance(variables=coords)
```

Likewise, the ```from_instance``` methods allows the domain to create an optimisation problem that can be solve using the definition of a particular instance.

```python
def from_instance(self, instance: Instance) -> TSP:
        n_nodes = len(instance) // 2
        coords = tuple([*zip(instance[::2], instance[1::2])])
        return TSP(nodes=n_nodes, coords=coords)
```

Finally, the ```extract_features``` methods are domain and user depedent since the features to extract may vary from everyone needs. For this example the methods looks like:

```python
    def extract_features(self, instance: Instance) -> tuple:
        """Extract the features of the instance based on the TSP domain.
           For the TSP the features are:
            - Size
            - Standard deviation of the distances
            - Centroid coordinates
            - Radius of the instance
            - Fraction of distinct instances
            - Rectangular area
            - Variance of the normalised nearest neighbours distances
            - Coefficient of variantion of the nearest neighbours distances
            - Cluster ratio
            - Mean cluster radius
        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Tuple[float]: Values of each feature
        """

        tsp = self.from_instance(instance)
        xs = instance[0::2]
        ys = instance[1::2]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        std_distances = np.std(tsp._distances)
        centroid = (np.mean(xs), np.mean(ys))  # (0.01 * np.sum(xs), 0.01 * np.sum(ys))

        centroid_distance = [np.linalg.norm(city - centroid) for city in tsp._coords]
        radius = np.mean(centroid_distance)

        fraction = len(np.unique(tsp._distances)) / (len(tsp._distances) / 2)
        # Top five only
        norm_distances = np.sort(tsp._distances)[::-1][:5] / np.max(tsp._distances)

        variance_nnds = np.var(norm_distances)
        variation_nnds = variance_nnds / np.mean(norm_distances)

        dbscan = DBSCAN()
        dbscan.fit(tsp._coords)
        cluster_ratio = len(set(dbscan.labels_)) / self.dimension
        # Cluster radius
        mean_cluster_radius = 0.0
        for label_id in dbscan.labels_:
            points_in_cluster = tsp._coords[dbscan.labels_ == label_id]
            cluster_centroid = (
                np.mean(points_in_cluster[:, 0]),
                np.mean(points_in_cluster[:, 1]),
            )
            mean_cluster_radius = np.mean(
                [np.linalg.norm(city - cluster_centroid) for city in tsp._coords]
            )
        mean_cluster_radius /= len(set(dbscan.labels_))

        return (
            self.dimension,
            std_distances,
            centroid[0],
            centroid[1],
            radius,
            fraction,
            area,
            variance_nnds,
            variation_nnds,
            cluster_ratio,
            mean_cluster_radius,
        )

    def extract_features_as_dict(self, instance: Instance) -> Mapping[str, float]:
        """Creates a dictionary with the features of the instance.
        The key are the names of each feature and the values are
        the values extracted from instance.

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Mapping[str, float]: Dictionary with the names/values of each feature
        """
        names = "size,std_distances,centroid_x,centroid_y,radius,fraction_distances,area,variance_nnNds,variation_nnNds,cluster_ratio,mean_cluster_radius"
        features = self.extract_features(instance)
        return {k: v for k, v in zip(names.split(","), features)}

```

The complete TSPDomain class looks like:

```python
class TSPDomain(Domain):
    """Domain to generate instances for the Symmetric Travelling Salesman Problem."""

    def __init__(
        self,
        dimension: int = 100,
        x_range: Tuple[int, int] = (0, 1000),
        y_range: Tuple[int, int] = (0, 1000),
        seed: int = 42,
    ):
        """Creates a new TSPDomain to generate instances for the Symmetric Travelling Salesman Problem

        Args:
            dimension (int, optional): Dimension of the instances to generate. Defaults to 100.
            x_range (Tuple[int, int], optional): Ranges for the Xs coordinates of each node/city. Defaults to (0, 1000).
            y_range (Tuple[int, int], optional): Ranges for the ys coordinates of each node/city. Defaults to (0, 1000).

        Raises:
            ValueError: If dimension is < 0
            ValueError: If x_range OR y_range does not have 2 dimensions each
            ValueError: If minimum ranges are greater than maximum ranges
        """
        if dimension < 0:
            raise ValueError(f"Expected dimension > 0 got {dimension}")
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError(
                f"Expected x_range and y_range to be a tuple with only to integers. Got: x_range = {x_range} and y_range = {y_range}"
            )
        x_min, x_max = x_range
        y_min, y_max = y_range
        if x_min < 0 or x_max <= x_min:
            raise ValueError(
                f"Expected x_range to be (x_min, x_max) where x_min >= 0 and x_max > x_min. Got: x_range {x_range}"
            )
        if y_min < 0 or y_max <= y_min:
            raise ValueError(
                f"Expected y_range to be (y_min, y_max) where y_min >= 0 and y_max > y_min. Got: y_range {y_range}"
            )

        self._x_range = x_range
        self._y_range = y_range
        __bounds = [
            (x_min, x_max) if i % 2 == 0 else (y_min, y_max)
            for i in range(dimension * 2)
        ]

        super().__init__(dimension=dimension, bounds=__bounds, name="TSP", seed=seed)

    def generate_instance(self) -> Instance:
        """Generates a new instances for the TSP domain

        Returns:
            Instance: New randomly generated instance
        """
        coords = self._rng.integers(
            low=(self._x_range[0], self._y_range[0]),
            high=(self._x_range[1], self._y_range[1]),
            size=(self.dimension, 2),
            dtype=int,
        )
        coords = coords.flatten()
        return Instance(coords)

    def extract_features(self, instance: Instance) -> tuple:
        """Extract the features of the instance based on the TSP domain.
           For the TSP the features are:
            - Size
            - Standard deviation of the distances
            - Centroid coordinates
            - Radius of the instance
            - Fraction of distinct instances
            - Rectangular area
            - Variance of the normalised nearest neighbours distances
            - Coefficient of variantion of the nearest neighbours distances
            - Cluster ratio
            - Mean cluster radius
        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Tuple[float]: Values of each feature
        """

        tsp = self.from_instance(instance)
        xs = instance[0::2]
        ys = instance[1::2]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        std_distances = np.std(tsp._distances)
        centroid = (np.mean(xs), np.mean(ys))  # (0.01 * np.sum(xs), 0.01 * np.sum(ys))

        centroid_distance = [np.linalg.norm(city - centroid) for city in tsp._coords]
        radius = np.mean(centroid_distance)

        fraction = len(np.unique(tsp._distances)) / (len(tsp._distances) / 2)
        # Top five only
        norm_distances = np.sort(tsp._distances)[::-1][:5] / np.max(tsp._distances)

        variance_nnds = np.var(norm_distances)
        variation_nnds = variance_nnds / np.mean(norm_distances)

        dbscan = DBSCAN()
        dbscan.fit(tsp._coords)
        cluster_ratio = len(set(dbscan.labels_)) / self.dimension
        # Cluster radius
        mean_cluster_radius = 0.0
        for label_id in dbscan.labels_:
            points_in_cluster = tsp._coords[dbscan.labels_ == label_id]
            cluster_centroid = (
                np.mean(points_in_cluster[:, 0]),
                np.mean(points_in_cluster[:, 1]),
            )
            mean_cluster_radius = np.mean(
                [np.linalg.norm(city - cluster_centroid) for city in tsp._coords]
            )
        mean_cluster_radius /= len(set(dbscan.labels_))

        return (
            self.dimension,
            std_distances,
            centroid[0],
            centroid[1],
            radius,
            fraction,
            area,
            variance_nnds,
            variation_nnds,
            cluster_ratio,
            mean_cluster_radius,
        )

    def extract_features_as_dict(self, instance: Instance) -> Mapping[str, float]:
        """Creates a dictionary with the features of the instance.
        The key are the names of each feature and the values are
        the values extracted from instance.

        Args:
            instance (Instance): Instance to extract the features from

        Returns:
            Mapping[str, float]: Dictionary with the names/values of each feature
        """
        names = "size,std_distances,centroid_x,centroid_y,radius,fraction_distances,area,variance_nnNds,variation_nnNds,cluster_ratio,mean_cluster_radius"
        features = self.extract_features(instance)
        return {k: v for k, v in zip(names.split(","), features)}

    def from_instance(self, instance: Instance) -> TSP:
        n_nodes = len(instance) // 2
        coords = tuple([*zip(instance[::2], instance[1::2])])
        return TSP(nodes=n_nodes, coords=coords)

```

## How to

1. [Run an experiment](01_eig_example.md)
2. [Create a domain](02_create_domain.md)
3. [Create a solver](03_create_algorithm.md)