# DIGNEApy Extension Guide: Adding Custom Generators, Archives, and Domains

This guide shows **exact code examples** for extending DIGNEApy with new components using the plugin architecture.

---

## Part 1: Creating a Custom Generator

### Example: Adding PSO-based QD Generator

**File**: `digneapy/generators/pso_generator.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Particle Swarm Optimization based Quality-Diversity generator.
"""

import numpy as np
from typing import Optional, Sequence
from collections.abc import Iterable

from digneapy._core import Instance, Domain, SupportsSolve, P
from digneapy.archives import Archive
from digneapy.operators import Selection, Replacement, binary_tournament_selection
from digneapy._core.scores import PerformanceFn, max_gap_target

from ._base_generator import BaseGenerator, GenResult
from ._registry import GeneratorRegistry


@GeneratorRegistry.register("pso-qd")
class PSOQDGenerator(BaseGenerator):
    """PSO-based Quality-Diversity Generator.
    
    Combines particle swarm optimization with novelty search to balance
    exploration (novelty) and exploitation (performance).
    
    Key features:
    - Position = instance variables
    - Velocity = mutation magnitude
    - Fitness = blend of novelty + performance
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        archive: Archive,
        pop_size: int = 100,
        generations: int = 1000,
        inertia: float = 0.7,
        cognition: float = 1.5,  # c1
        social: float = 1.5,     # c2
        phi: float = 0.85,
        seed: int = 42,
    ):
        """Initialize PSO-QD Generator.
        
        Args:
            domain: Problem domain
            portfolio: Solver portfolio
            archive: Archive for novelty tracking
            pop_size: Population size
            generations: Number of generations
            inertia: PSO inertia weight
            cognition: Cognitive parameter (c1)
            social: Social parameter (c2)
            phi: Novelty/performance blending weight
            seed: Random seed
        """
        super().__init__(
            domain=domain,
            portfolio=portfolio,
            generations=generations,
            seed=seed,
        )

        self.archive = archive
        self.pop_size = pop_size
        self.inertia = inertia
        self.cognition = cognition
        self.social = social
        self.phi = phi

        # PSO state
        self.population = None
        self.velocities = None
        self.best_personal = None
        self.best_personal_fitness = None
        self.best_global = None
        self.best_global_fitness = -np.inf

    def __str__(self) -> str:
        return (
            f"PSOQDGenerator("
            f"pop_size={self.pop_size},"
            f"gen={self.generations},"
            f"inertia={self.inertia},"
            f"phi={self.phi})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def _generate_population(self) -> Sequence[Instance]:
        """Generate initial population randomly."""
        return [self.domain.generate_instance() for _ in range(self.pop_size)]

    def _initialize_pso_state(self, population: Sequence[Instance]):
        """Initialize PSO velocities and best positions."""
        self.population = population
        n_vars = len(population[0].variables)

        # Initialize velocities (small random values)
        self.velocities = self.rng.uniform(
            -0.1, 0.1, size=(self.pop_size, n_vars)
        )

        # Track personal bests
        self.best_personal = [instance for instance in population]
        self.best_personal_fitness = np.full(self.pop_size, -np.inf)

    def _update_particles(self, fitness_scores: np.ndarray) -> Sequence[Instance]:
        """Update particle positions and velocities using PSO equations.
        
        PSO update rule:
            v_i(t+1) = w*v_i(t) + c1*rand()*(x_best_i - x_i(t)) 
                                 + c2*rand()*(x_best_global - x_i(t))
            x_i(t+1) = x_i(t) + v_i(t+1)
        """
        new_population = []

        for i in range(self.pop_size):
            # Update fitness tracking
            if fitness_scores[i] > self.best_personal_fitness[i]:
                self.best_personal_fitness[i] = fitness_scores[i]
                self.best_personal[i] = self.population[i]

            if fitness_scores[i] > self.best_global_fitness:
                self.best_global_fitness = fitness_scores[i]
                self.best_global = self.population[i]

            # PSO velocity update
            r1 = self.rng.uniform(0, 1)
            r2 = self.rng.uniform(0, 1)

            personal_best_vars = np.array(self.best_personal[i].variables)
            current_vars = np.array(self.population[i].variables)
            global_best_vars = np.array(self.best_global.variables if self.best_global else current_vars)

            self.velocities[i] = (
                self.inertia * self.velocities[i]
                + self.cognition * r1 * (personal_best_vars - current_vars)
                + self.social * r2 * (global_best_vars - current_vars)
            )

            # Apply velocity bounds
            self.velocities[i] = np.clip(self.velocities[i], -0.5, 0.5)

            # Update position
            new_vars = np.array(self.population[i].variables) + self.velocities[i]
            new_vars = np.clip(new_vars, 0, 1)  # Assume normalized variables

            # Create new instance
            new_instance = Instance(
                variables=new_vars.tolist(),
                s=0.0,
            )
            new_population.append(new_instance)

        return new_population

    def __call__(self, verbose: bool = False) -> GenResult:
        """Execute PSO-QD algorithm."""
        from digneapy._core import NS

        # Initialize
        population = self._generate_population()
        self._initialize_pso_state(population)

        novelty_search = NS(archive=self.archive)

        best_instance = None
        best_performance = -np.inf

        # Evolution loop
        for gen_idx in range(self.generations):
            # Evaluate population
            performances, descriptors = self._evaluate_population(population)

            # Compute novelty scores
            novelty_scores = novelty_search(descriptors)

            # Add novel solutions to archive
            self.archive.extend(population, novelty_scores, descriptors)

            # Compute combined fitness
            fitness_scores = self._compute_fitness(performances, novelty_scores, self.phi)

            # Track best
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_performance:
                best_performance = fitness_scores[best_idx]
                best_instance = population[best_idx]

            # PSO update
            population = self._update_particles(fitness_scores)

            # Logging
            stats = {
                'generation': gen_idx,
                'pop_size': len(population),
                'best_fitness': np.max(fitness_scores),
                'mean_fitness': np.mean(fitness_scores),
                'archive_size': len(self.archive),
            }
            self.logbook.append(stats)

            if verbose and (gen_idx + 1) % 10 == 0:
                print(
                    f"Gen {gen_idx + 1}: "
                    f"best_fit={stats['best_fitness']:.4f}, "
                    f"archive_size={stats['archive_size']}"
                )

        return GenResult(
            target="PSO-QD",
            instances=np.array(self.archive.instances),
            history=self.logbook,
        )
```

**Register in** `digneapy/generators/__init__.py`:

```python
from .pso_generator import PSOQDGenerator

__all__ = ["PSOQDGenerator", ...]
```

---

## Part 2: Creating a Custom Archive

### Example: Adding k-NN Archive

**File**: `digneapy/archives/knn_archive.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
k-Nearest Neighbors Archive for Quality-Diversity algorithms.
Stores diverse solutions based on descriptor similarity.
"""

from typing import Sequence, Optional, Tuple
import numpy as np
from sklearn.neighbors import KDTree

from digneapy._core import Instance
from ._base_archive import Archive
from ._registry import ArchiveRegistry


@ArchiveRegistry.register("knn")
class KNNArchive(Archive):
    """Archive using k-NN for diversity maintenance.
    
    Key idea: Only store solutions that are distant from their
    k nearest neighbors in descriptor space.
    
    Features:
    - Efficient kD-tree queries
    - Automatic diversity through k-NN distance
    - No fixed grid structure needed
    """

    def __init__(
        self,
        threshold: float = 0.5,
        k: int = 5,
        instances: Optional[Sequence[Instance]] = None,
        dtype=np.float64,
    ):
        """Initialize k-NN Archive.
        
        Args:
            threshold: Minimum k-NN distance to include solution
            k: Number of neighbors to consider
            instances: Initial instances
            dtype: Data type for arrays
        """
        super().__init__(threshold=threshold, instances=instances, dtype=dtype)
        self._k = k
        self._kdtree = None

    @property
    def k(self) -> int:
        return self._k

    def _rebuild_kdtree(self):
        """Rebuild KD-tree when instances change."""
        if len(self._storage["descriptors"]) > 0:
            self._kdtree = KDTree(
                np.array(self._storage["descriptors"]),
                metric="euclidean"
            )
        else:
            self._kdtree = None

    def extend(
        self,
        instances: Sequence[Instance],
        novelty_scores: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None,
    ):
        """Add instances with k-NN diversity check.
        
        Only adds instances if their minimum distance to k nearest
        neighbors is >= threshold.
        """
        if descriptors is None:
            descriptors = np.array([inst.descriptor for inst in instances])

        # Compute k-NN distances for each new instance
        knn_distances = self._compute_knn_distances(descriptors)

        # Only keep instances with sufficient diversity
        to_add = np.where(knn_distances >= self.threshold)[0]

        self._storage["instances"].extend([instances[i] for i in to_add])
        self._storage["descriptors"].extend(descriptors[to_add])

        # Rebuild tree after adding
        self._rebuild_kdtree()

    def _compute_knn_distances(
        self,
        new_descriptors: np.ndarray,
    ) -> np.ndarray:
        """Compute minimum distance to k nearest neighbors.
        
        Args:
            new_descriptors: (n_new, n_dims) array
        
        Returns:
            (n_new,) array of minimum distances
        """
        if self._kdtree is None:
            # Archive is empty, all distances are infinite
            return np.full(len(new_descriptors), np.inf)

        # Query k+1 neighbors (including self if in archive)
        distances, indices = self._kdtree.query(
            new_descriptors,
            k=min(self._k, len(self._storage["instances"]))
        )

        # Minimum distance to k-NN (excluding self)
        if distances.ndim == 1:
            return distances
        return np.min(distances, axis=1)

    def add_to_cell(self, instance: Instance, descriptor: Optional[np.ndarray] = None):
        """k-NN archives don't have cells, just add instance."""
        if descriptor is None:
            descriptor = instance.descriptor

        self._storage["instances"].append(instance)
        self._storage["descriptors"].append(descriptor)
        self._rebuild_kdtree()

    def index_of(self, descriptors: np.ndarray) -> np.ndarray:
        """Return indices of k nearest neighbors for descriptors.
        
        Args:
            descriptors: (batch_size, n_dims) array
        
        Returns:
            (batch_size,) array of nearest neighbor indices
        """
        if self._kdtree is None:
            raise ValueError("Archive is empty")

        _, indices = self._kdtree.query(descriptors, k=1)
        if indices.ndim == 2:
            return indices[:, 0]
        return indices

    def asdict(self) -> dict:
        return {
            "type": "KNNArchive",
            "threshold": self._threshold,
            "k": self._k,
            "instances": {
                i: inst.asdict()
                for i, inst in enumerate(self._storage["instances"])
            },
        }

    def __str__(self) -> str:
        return f"KNNArchive(k={self._k}, threshold={self._threshold}, size={len(self)})"

    def __repr__(self) -> str:
        return self.__str__()
```

---

## Part 3: Creating a Custom Domain

### Example: Adding Traveling Salesman Problem (TSP) Domain

**File**: `digneapy/domains/custom_tsp.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Custom TSP Domain Example - demonstrates domain implementation.
"""

from typing import Sequence
import numpy as np

from digneapy._core import Instance, SupportsSolve, P
from digneapy.solvers import TSPSolver, GreedyTSPSolver
from ._domain_protocol import DomainProtocol
from ._registry import DomainRegistry


@DomainRegistry.register("custom-tsp")
class CustomTSPDomain:
    """Custom Traveling Salesman Problem domain.
    
    Features:
    - 2D point cloud generation
    - City coordinate mutation
    - Edge-based crossover
    - Multiple solvers (greedy, nearest-neighbor)
    """

    def __init__(self, n_cities: int = 50, bounds: Tuple[float, float] = (0, 100)):
        """Initialize TSP domain.
        
        Args:
            n_cities: Number of cities
            bounds: Coordinate bounds (min, max)
        """
        self.n_cities = n_cities
        self.bounds = bounds
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        return f"CustomTSP-{self.n_cities}"

    @property
    def features_size(self) -> int:
        """Descriptor features: centroid, spread, cluster density."""
        return 5

    def generate_instance(self) -> Instance:
        """Generate random TSP instance (point cloud)."""
        # Random city coordinates
        cities = self._rng.uniform(
            self.bounds[0],
            self.bounds[1],
            size=(self.n_cities, 2)
        )
        return Instance(variables=cities.flatten().tolist(), s=0.0)

    def mutate(self, instance: Instance) -> Instance:
        """Mutate instance by slightly moving cities."""
        cities = np.array(instance.variables).reshape(-1, 2)

        # Gaussian mutation on city coordinates
        mutation = self._rng.normal(0, 2.0, size=cities.shape)
        cities = np.clip(
            cities + mutation,
            self.bounds[0],
            self.bounds[1]
        )

        return Instance(variables=cities.flatten().tolist(), s=0.0)

    def crossover(self, parent1: Instance, parent2: Instance) -> Instance:
        """Crossover: blend city coordinates from both parents."""
        cities1 = np.array(parent1.variables).reshape(-1, 2)
        cities2 = np.array(parent2.variables).reshape(-1, 2)

        # Random blending
        alpha = self._rng.uniform(0.3, 0.7)
        offspring = alpha * cities1 + (1 - alpha) * cities2

        return Instance(variables=offspring.flatten().tolist(), s=0.0)

    def get_solver(self, solver_name: str) -> SupportsSolve[P]:
        """Get solver by name."""
        if solver_name == "greedy":
            return GreedyTSPSolver()
        elif solver_name == "nearest-neighbor":
            return NearestNeighborTSPSolver()
        else:
            raise ValueError(f"Unknown TSP solver: {solver_name}")

    def get_default_solvers(self) -> Sequence[SupportsSolve[P]]:
        """Return default solver portfolio."""
        return [
            GreedyTSPSolver(),
            NearestNeighborTSPSolver(),
        ]

    def get_features(self, instance: Instance) -> np.ndarray:
        """Extract features for descriptor computation.
        
        Features:
        1. Centroid X
        2. Centroid Y
        3. Distance spread
        4. Cluster compactness
        5. Convex hull area ratio
        """
        cities = np.array(instance.variables).reshape(-1, 2)

        # Centroid
        centroid = np.mean(cities, axis=0)

        # Spread
        spread = np.mean(np.linalg.norm(cities - centroid, axis=1))

        # Compactness (variance)
        compactness = np.mean(np.var(cities, axis=0))

        # Convex hull (simplified: range)
        x_range = np.max(cities[:, 0]) - np.min(cities[:, 0])
        y_range = np.max(cities[:, 1]) - np.min(cities[:, 1])
        hull_ratio = (x_range * y_range) / (100 ** 2)  # Normalized to bounds

        return np.array([
            centroid[0],
            centroid[1],
            spread,
            compactness,
            hull_ratio,
        ])
```

---

## Part 4: Using Custom Components

### Quick Start Example

```python
from digneapy import GeneratorConfig, GeneratorRegistry

# Create config with custom PSO-QD generator
config = GeneratorConfig(
    generator_type="pso-qd",    # ✅ Our custom generator!
    domain_type="custom-tsp",   # ✅ Our custom domain!
    archive_type="knn",         # ✅ Our custom archive!
    generations=100,
    population_size=50,
    domain_params={"n_cities": 50},
    archive_params={"k": 5, "threshold": 0.5},
    generator_params={"inertia": 0.7, "phi": 0.85},
)

# Create and run
gen = config.create()
results = gen(verbose=True)

print(f"Generated {len(results.instances)} instances")
print(f"Registered generators: {GeneratorRegistry.list_generators()}")
# Output: ['dea', 'ea', 'map-elites', 'pso-qd']
```

---

## Part 5: Testing Custom Components

### Test Template

```python
# tests/test_custom_pso_generator.py
import pytest
import numpy as np
from digneapy import GeneratorRegistry
from digneapy.domains import KnapsackDomain
from digneapy.archives import CVTArchive
from digneapy.generators import PSOQDGenerator


class TestPSOQDGenerator:
    """Test custom PSO-QD generator."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        domain = KnapsackDomain(n_items=20)
        archive = CVTArchive(k=10, ranges=[(-1, 1), (-1, 1)], n_samples=100)
        portfolio = domain.get_default_solvers()

        return {
            'domain': domain,
            'archive': archive,
            'portfolio': portfolio,
        }

    def test_generator_registered(self):
        """Verify generator is registered."""
        assert GeneratorRegistry.is_registered("pso-qd")
        assert "pso-qd" in GeneratorRegistry.list_generators()

    def test_create_from_registry(self, setup):
        """Verify creation from registry."""
        gen = GeneratorRegistry.create(
            "pso-qd",
            **setup,
            pop_size=50,
            generations=10,
        )
        assert isinstance(gen, PSOQDGenerator)

    def test_generator_execution(self, setup):
        """Test generator produces results."""
        gen = PSOQDGenerator(
            **setup,
            pop_size=10,
            generations=5,
        )

        results = gen(verbose=False)

        assert results.instances is not None
        assert len(results.instances) > 0
        assert results.history is not None
        assert len(results.history) == 5  # 5 generations logged

    def test_generator_str(self, setup):
        """Test string representation."""
        gen = PSOQDGenerator(**setup, pop_size=50, generations=100)
        assert "PSOQDGenerator" in str(gen)
        assert "pop_size=50" in str(gen)
```

---

## Summary: Component Registration Flow

```
1. Create component class (inherits from Base or implements Protocol)
   └─ Add @Registry.register("name") decorator

2. Components auto-register when imported
   └─ Register in __init__.py

3. Create via Registry or Config
   └─ Dynamic instantiation with discovery

4. Use in generators/experiments
   └─ Seamless integration

5. Share with community
   └─ Plugin pattern enables distribution
```

---

## Distribution: Publishing Custom Components

### Setup for pip distribution

**File**: `setup.py` (in your custom component package)

```python
setup(
    name="digneapy-custom-pso",
    version="1.0.0",
    install_requires=["digneapy>=1.0.0"],
    entry_points={
        "digneapy.generators": [
            "pso-qd = digneapy_custom_pso:PSOQDGenerator",
        ],
    },
)
```

Users can then `pip install digneapy-custom-pso` and use your generator!
