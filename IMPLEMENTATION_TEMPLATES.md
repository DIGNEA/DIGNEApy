# DIGNEApy Extensible Architecture: Implementation Templates

This file contains ready-to-use code snippets for implementing the plugin-based architecture.

## Template 1: BaseGenerator Class

**File**: `digneapy/generators/_base_generator.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Base generator class for all Quality-Diversity algorithms.
Handles common evaluation, descriptor computation, and result tracking.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass

from digneapy._core import RNG, Domain, Instance, SupportsSolve, P
from digneapy._core._metrics import Logbook, Statistics
from digneapy._core.descriptors import DESCRIPTORS
from digneapy._core.scores import PerformanceFn, max_gap_target
from digneapy.archives import Archive


@dataclass
class GenResult:
    """Results from a generator run.
    
    Attributes:
        target (str): Name of target solver
        instances (np.ndarray): Generated instances
        history (Logbook): Evolution history
        metrics (Optional[pd.Series]): Performance metrics
    """
    target: str
    instances: np.ndarray
    history: Logbook
    metrics: Optional['pd.Series'] = None


class BaseGenerator(ABC, RNG):
    """Abstract base class for all QD generators.
    
    Provides:
    - Population evaluation pipeline
    - Descriptor computation
    - Result formatting
    - Statistics tracking
    
    Subclasses must implement:
    - _generate_population()  : Create initial population
    - __call__()              : Main algorithm loop
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        generations: int = 1000,
        repetitions: int = 1,
        descriptor_strategy: str = "features",
        performance_function: PerformanceFn = max_gap_target,
        seed: int = 42,
    ):
        """Initialize base generator.
        
        Args:
            domain: Problem domain (BPP, KP, TSP, etc.)
            portfolio: Solvers to evaluate instances against
            generations: Number of generations to evolve
            repetitions: Number of times to run each solver
            descriptor_strategy: How to compute descriptors ("features", "pca", etc.)
            performance_function: Function to compute performance scores
            seed: Random seed
        """
        RNG.__init__(self, seed)
        self.domain = domain
        self.portfolio = list(portfolio)
        self.generations = generations
        self.repetitions = repetitions
        self.descriptor_strategy = descriptor_strategy
        self.performance_function = performance_function
        self.logbook = Logbook()
        
        # Verify descriptor strategy exists
        if descriptor_strategy not in DESCRIPTORS:
            raise ValueError(
                f"Descriptor '{descriptor_strategy}' not found. "
                f"Available: {list(DESCRIPTORS.keys())}"
            )

    def _evaluate_population(
        self,
        population: Sequence[Instance],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate population against portfolio.
        
        SHARED LOGIC - previously duplicated in EAGenerator and MapElitesGenerator
        
        Args:
            population: Instances to evaluate
        
        Returns:
            (performance_scores, diversity_scores)
                performance_scores: (n_instances,) array of performance values
                diversity_scores: (n_instances, descriptor_dims) array of descriptors
        """
        n_instances = len(population)
        performances = np.zeros(n_instances, dtype=np.float64)
        
        # Evaluate each instance
        for instance_idx, instance in enumerate(population):
            instance_performance = []
            
            for solver in self.portfolio:
                for _ in range(self.repetitions):
                    result = solver.solve(instance)
                    instance_performance.append(result.best_obj)
            
            # Aggregate performance across solvers and repetitions
            performances[instance_idx] = self.performance_function(
                np.array(instance_performance)
            )
        
        # Compute descriptors
        descriptors = self._update_descriptors(population)[0]
        
        return performances, descriptors

    def _update_descriptors(
        self,
        population: np.ndarray | Sequence[Instance],
        portfolio_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute descriptors for instances.
        
        Supports multiple descriptor strategies (features, PCA, autoencoder, etc.)
        
        Args:
            population: Instances or their features
            portfolio_scores: Optional pre-computed performance scores
        
        Returns:
            (descriptors, portfolio_scores)
        """
        # Convert to instances if needed
        if not isinstance(population[0], Instance):
            instances = population
        else:
            instances = population
        
        # Get descriptor function for strategy
        descriptor_fn = DESCRIPTORS[self.descriptor_strategy]
        
        # Compute descriptors
        descriptors = np.array([
            descriptor_fn(instance) for instance in instances
        ])
        
        return descriptors, portfolio_scores

    def _compute_fitness(
        self,
        performance_scores: np.ndarray,
        novelty_scores: np.ndarray,
        phi: float = 0.85,
    ) -> np.ndarray:
        """Compute fitness as blend of performance and novelty.
        
        Formula: fitness = phi * novelty + (1-phi) * normalized_performance
        
        Args:
            performance_scores: Raw performance values
            novelty_scores: Novelty/sparseness scores
            phi: Blending weight (0.0 = pure performance, 1.0 = pure novelty)
        
        Returns:
            Combined fitness scores
        """
        # Normalize performance to [0, 1]
        perf_min = np.min(performance_scores)
        perf_max = np.max(performance_scores)
        if perf_max > perf_min:
            perf_normalized = (performance_scores - perf_min) / (perf_max - perf_min)
        else:
            perf_normalized = np.ones_like(performance_scores)
        
        # Combined fitness
        fitness = phi * novelty_scores + (1.0 - phi) * perf_normalized
        return fitness

    @abstractmethod
    def __call__(self, verbose: bool = False) -> GenResult:
        """Execute the generator algorithm.
        
        Must return GenResult with generated instances and history.
        """
        pass

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"domain={self.domain},"
            f"generations={self.generations},"
            f"portfolio_size={len(self.portfolio)}"
            f")"
        )

    def __repr__(self) -> str:
        return self.__str__()
```

---

## Template 2: ArchiveProtocol

**File**: `digneapy/archives/_archive_protocol.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Protocol definition for archives - enables loose coupling and custom implementations.
"""

from typing import Protocol, Sequence, Optional
import numpy as np
from digneapy._core import Instance


class ArchiveProtocol(Protocol):
    """Interface that all archives must implement.
    
    This protocol enables composition over inheritance and removes
    isinstance() type checks from generators.
    
    Any class implementing these methods can be used as an archive.
    """

    @property
    def instances(self) -> Sequence[Instance]:
        """Return all stored instances.
        
        Returns:
            Sequence of Instance objects
        """
        ...

    @property
    def descriptors(self) -> np.ndarray:
        """Return descriptors of all instances.
        
        Returns:
            (n_instances, n_dims) array of descriptor vectors
        """
        ...

    @property
    def threshold(self) -> float:
        """Return novelty/fitness threshold for inclusion.
        
        Returns:
            Threshold value
        """
        ...

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set novelty threshold."""
        ...

    def extend(
        self,
        instances: Sequence[Instance],
        novelty_scores: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None,
    ) -> None:
        """Add instances to archive if novelty_scores >= threshold.
        
        Args:
            instances: Instances to potentially add
            novelty_scores: Novelty scores (computed if not provided)
            descriptors: Pre-computed descriptors (computed if not provided)
        """
        ...

    def add_to_cell(
        self,
        instance: Instance,
        descriptor: Optional[np.ndarray] = None,
    ) -> None:
        """Archive-specific: Add instance to cell/region.
        
        For grid/CVT archives: places instance in appropriate cell.
        For unstructured archives: can be no-op or used for indexing.
        
        Args:
            instance: Instance to add
            descriptor: Pre-computed descriptor
        """
        ...

    def index_of(self, descriptors: np.ndarray) -> np.ndarray:
        """Return cell/region indices for descriptors.
        
        Args:
            descriptors: (batch_size, n_dims) array of descriptors
        
        Returns:
            (batch_size,) array of integer indices
        """
        ...

    def asdict(self) -> dict:
        """Serialize archive to dictionary for saving.
        
        Returns:
            Dictionary representation of archive
        """
        ...

    def __len__(self) -> int:
        """Return number of instances in archive."""
        ...

    def __iter__(self):
        """Iterate over instances in archive."""
        ...

    def __bool__(self) -> bool:
        """Return True if archive is non-empty."""
        ...
```

---

## Template 3: DomainProtocol

**File**: `digneapy/_core/domain_protocol.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Protocol definition for problem domains - enables swappable domains.
"""

from typing import Protocol, Sequence
import numpy as np
from . import Instance, P, SupportsSolve


class DomainProtocol(Protocol):
    """Interface that all problem domains must implement.
    
    A domain encapsulates:
    - Problem-specific instance generation
    - Domain-specific mutation/crossover
    - Solver portfolio for that domain
    """

    @property
    def name(self) -> str:
        """Domain name for identification.
        
        Examples: "BPP", "KP", "TSP", "VRP"
        """
        ...

    @property
    def features_size(self) -> int:
        """Number of features for descriptor computation.
        
        Returns:
            Number of features returned by get_features()
        """
        ...

    def generate_instance(self) -> Instance:
        """Generate a random instance in this domain.
        
        Called during initial population generation.
        
        Returns:
            Random Instance with valid decision variables
        """
        ...

    def mutate(self, instance: Instance) -> Instance:
        """Apply domain-specific mutation to instance.
        
        Args:
            instance: Instance to mutate
        
        Returns:
            Mutated copy of instance
        """
        ...

    def crossover(self, parent1: Instance, parent2: Instance) -> Instance:
        """Apply domain-specific crossover to create offspring.
        
        Args:
            parent1: First parent instance
            parent2: Second parent instance
        
        Returns:
            Offspring instance combining both parents
        """
        ...

    def get_solver(self, solver_name: str) -> SupportsSolve[P]:
        """Get a solver by name for this domain.
        
        Args:
            solver_name: Identifier for solver ("greedy", "genetic", etc.)
        
        Returns:
            Solver instance implementing SupportsSolve protocol
        
        Raises:
            ValueError: If solver_name not found
        """
        ...

    def get_default_solvers(self) -> Sequence[SupportsSolve[P]]:
        """Get default solver portfolio for this domain.
        
        Returns:
            Sequence of solver instances
        """
        ...

    def get_features(self, instance: Instance) -> np.ndarray:
        """Extract features from instance for descriptors.
        
        Args:
            instance: Instance to extract features from
        
        Returns:
            (features_size,) array of feature values
        """
        ...
```

---

## Template 4: Generator Registry

**File**: `digneapy/generators/_registry.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Registry for generator implementations - enables plugin discovery and instantiation.
"""

from typing import Type, Dict, Callable, Any
from ._base_generator import BaseGenerator


class GeneratorRegistry:
    """Central registry for all generator implementations.
    
    Features:
    - Dynamic registration of new generators
    - Factory instantiation
    - Plugin discovery
    - Type validation
    """

    _generators: Dict[str, Type[BaseGenerator]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a generator implementation.
        
        Usage:
            @GeneratorRegistry.register("ea")
            class EAGenerator(BaseGenerator):
                pass
        
        Args:
            name: Unique identifier for generator
        
        Returns:
            Decorator function
        """
        def decorator(generator_cls: Type[BaseGenerator]) -> Type[BaseGenerator]:
            if not issubclass(generator_cls, BaseGenerator):
                raise TypeError(
                    f"Generator {generator_cls.__name__} must inherit from BaseGenerator"
                )
            
            if name in cls._generators:
                raise ValueError(f"Generator '{name}' already registered")
            
            cls._generators[name] = generator_cls
            generator_cls._registry_name = name
            return generator_cls
        
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseGenerator:
        """Instantiate a registered generator.
        
        Usage:
            gen = GeneratorRegistry.create("ea",
                domain=my_domain,
                portfolio=solvers,
                generations=100
            )
        
        Args:
            name: Registered generator name
            **kwargs: Arguments to pass to generator constructor
        
        Returns:
            Generator instance
        
        Raises:
            ValueError: If generator not registered
        """
        if name not in cls._generators:
            available = ", ".join(cls._generators.keys())
            raise ValueError(
                f"Generator '{name}' not registered. "
                f"Available: [{available}]"
            )
        
        generator_cls = cls._generators[name]
        return generator_cls(**kwargs)

    @classmethod
    def list_generators(cls) -> list[str]:
        """List all registered generator names.
        
        Returns:
            List of generator identifiers
        """
        return sorted(list(cls._generators.keys()))

    @classmethod
    def get(cls, name: str) -> Type[BaseGenerator]:
        """Get generator class without instantiating.
        
        Args:
            name: Generator name
        
        Returns:
            Generator class
        
        Raises:
            ValueError: If not registered
        """
        if name not in cls._generators:
            raise ValueError(f"Generator '{name}' not registered")
        return cls._generators[name]

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a generator (mainly for testing).
        
        Args:
            name: Generator name to remove
        """
        if name in cls._generators:
            del cls._generators[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if generator is registered.
        
        Args:
            name: Generator name
        
        Returns:
            True if registered
        """
        return name in cls._generators
```

---

## Template 5: Archive Registry

**File**: `digneapy/archives/_registry.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Registry for archive implementations."""

from typing import Type, Dict, Callable, Protocol
import numpy as np
from digneapy._core import Instance


class ArchiveRegistry:
    """Central registry for archive implementations."""

    _archives: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register an archive implementation.
        
        Usage:
            @ArchiveRegistry.register("grid")
            class GridArchive:
                pass
        """
        def decorator(archive_cls: Type) -> Type:
            # Verify it has required methods of ArchiveProtocol
            required_methods = [
                'instances', 'descriptors', 'threshold', 'extend',
                'index_of', 'asdict', '__len__', '__iter__'
            ]
            for method in required_methods:
                if not hasattr(archive_cls, method):
                    raise TypeError(
                        f"Archive {archive_cls.__name__} missing required method: {method}"
                    )
            
            if name in cls._archives:
                raise ValueError(f"Archive '{name}' already registered")
            
            cls._archives[name] = archive_cls
            return archive_cls
        
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        """Instantiate a registered archive.
        
        Args:
            name: Registered archive name
            **kwargs: Arguments to pass to archive constructor
        
        Returns:
            Archive instance
        
        Raises:
            ValueError: If archive not registered
        """
        if name not in cls._archives:
            available = ", ".join(cls._archives.keys())
            raise ValueError(
                f"Archive '{name}' not registered. "
                f"Available: [{available}]"
            )
        
        archive_cls = cls._archives[name]
        return archive_cls(**kwargs)

    @classmethod
    def list_archives(cls) -> list[str]:
        """List all registered archive names."""
        return sorted(list(cls._archives.keys()))

    @classmethod
    def get(cls, name: str) -> Type:
        """Get archive class without instantiating."""
        if name not in cls._archives:
            raise ValueError(f"Archive '{name}' not registered")
        return cls._archives[name]
```

---

## Template 6: Domain Registry

**File**: `digneapy/domains/_registry.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Registry for domain implementations."""

from typing import Type, Dict, Callable


class DomainRegistry:
    """Central registry for problem domain implementations."""

    _domains: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a domain implementation."""
        def decorator(domain_cls: Type) -> Type:
            # Verify it has required methods of DomainProtocol
            required_attrs = [
                'name', 'features_size', 'generate_instance', 'mutate',
                'crossover', 'get_solver', 'get_default_solvers', 'get_features'
            ]
            for attr in required_attrs:
                if not (hasattr(domain_cls, attr) or hasattr(domain_cls(), attr)):
                    raise TypeError(
                        f"Domain {domain_cls.__name__} missing required attribute: {attr}"
                    )
            
            if name in cls._domains:
                raise ValueError(f"Domain '{name}' already registered")
            
            cls._domains[name] = domain_cls
            return domain_cls
        
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        """Instantiate a registered domain."""
        if name not in cls._domains:
            available = ", ".join(cls._domains.keys())
            raise ValueError(
                f"Domain '{name}' not registered. "
                f"Available: [{available}]"
            )
        
        domain_cls = cls._domains[name]
        return domain_cls(**kwargs)

    @classmethod
    def list_domains(cls) -> list[str]:
        """List all registered domain names."""
        return sorted(list(cls._domains.keys()))
```

---

## Template 7: Adding to __init__.py

**File**: `digneapy/__init__.py`

```python
# ... existing imports ...

# Import registries
from .generators._registry import GeneratorRegistry
from .archives._registry import ArchiveRegistry
from .domains._registry import DomainRegistry

# Import and register generators
from .generators._base_generator import BaseGenerator
from .generators import EAGenerator, MapElitesGenerator, DEAGenerator

# Import and register archives
from .archives import GridArchive, CVTArchive, Archive

# Import and register domains
from .domains import BinPackingDomain, KnapsackDomain, TSPDomain

# Make registries available
__all__ = [
    'GeneratorRegistry',
    'ArchiveRegistry',
    'DomainRegistry',
    # ... rest ...
]

# Verify all built-in components are registered
def _verify_registrations():
    """Ensure built-in implementations are registered."""
    assert GeneratorRegistry.is_registered("ea"), "EAGenerator not registered"
    assert GeneratorRegistry.is_registered("map-elites"), "MapElitesGenerator not registered"
    assert GeneratorRegistry.is_registered("dea"), "DEAGenerator not registered"
    
    assert ArchiveRegistry.is_registered("grid"), "GridArchive not registered"
    assert ArchiveRegistry.is_registered("cvt"), "CVTArchive not registered"
    
    assert DomainRegistry.is_registered("bpp"), "BinPackingDomain not registered"
    assert DomainRegistry.is_registered("kp"), "KnapsackDomain not registered"
    assert DomainRegistry.is_registered("tsp"), "TSPDomain not registered"

_verify_registrations()
```

---

## Template 8: Configuration Class

**File**: `digneapy/_core/config.py`

```python
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Configuration system for creating generators via composition."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from digneapy.generators import BaseGenerator


@dataclass
class GeneratorConfig:
    """Configuration for declarative generator creation.
    
    Usage:
        config = GeneratorConfig(
            generator_type="map-elites",
            domain_type="kp",
            archive_type="cvt",
            generations=50,
            population_size=100,
        )
        gen = config.create()
        results = gen()
    """

    # Component types
    generator_type: str     # "ea", "map-elites", "dea"
    domain_type: str        # "bpp", "kp", "tsp"
    archive_type: str       # "grid", "cvt"

    # Common parameters
    generations: int = 1000
    population_size: int = 100
    seed: int = 42

    # Optional parameters
    archive_params: Dict[str, Any] = field(default_factory=dict)
    domain_params: Dict[str, Any] = field(default_factory=dict)
    generator_params: Dict[str, Any] = field(default_factory=dict)

    def create(self) -> "BaseGenerator":
        """Create and return configured generator."""
        from digneapy.generators import GeneratorRegistry
        from digneapy.archives import ArchiveRegistry
        from digneapy.domains import DomainRegistry

        # Create domain
        domain = DomainRegistry.create(
            self.domain_type,
            **self.domain_params
        )

        # Create archive
        archive = ArchiveRegistry.create(
            self.archive_type,
            **self.archive_params
        )

        # Get default portfolio from domain
        portfolio = domain.get_default_solvers()

        # Create generator
        return GeneratorRegistry.create(
            self.generator_type,
            domain=domain,
            archive=archive,
            portfolio=portfolio,
            generations=self.generations,
            population_size=self.population_size,
            seed=self.seed,
            **self.generator_params
        )

    def __str__(self) -> str:
        return (
            f"GeneratorConfig("
            f"{self.generator_type}, {self.domain_type}, {self.archive_type}, "
            f"gen={self.generations}, pop={self.population_size})"
        )
```

---

## Example: Using Templates

```python
# Old API (still works)
from digneapy import EAGenerator, GridArchive
from digneapy.domains import KnapsackDomain

domain = KnapsackDomain(n_items=50)
archive = GridArchive(dimensions=(10, 10), ranges=[(-1, 1), (-1, 1)])
portfolio = domain.get_default_solvers()

gen = EAGenerator(
    domain=domain,
    archive=archive,
    portfolio=portfolio,
    generations=100,
)
results = gen(verbose=True)

# New declarative API
from digneapy import GeneratorConfig

config = GeneratorConfig(
    generator_type="ea",
    domain_type="kp",
    archive_type="grid",
    generations=100,
    population_size=50,
    domain_params={"n_items": 50},
    archive_params={"dimensions": (10, 10), "ranges": [(-1, 1), (-1, 1)]},
)

gen = config.create()
results = gen(verbose=True)

# Registry API for dynamic creation
from digneapy import GeneratorRegistry

# Discover available generators
print(GeneratorRegistry.list_generators())  # ['dea', 'ea', 'map-elites']

# Create by name
gen = GeneratorRegistry.create("map-elites", ...)
```
