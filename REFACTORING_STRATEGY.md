# DIGNEApy Refactoring Strategy: Plugin-Based Architecture

## Overview

This guide provides a systematic approach to refactor DIGNEApy into a **plugin-based architecture** for easy integration of new generators, archives, and domains—inspired by Pyribs' modularity but maintaining your eager evaluation model (no Ask-Tell pattern).

---

## 1. Core Architecture Redesign

### Before: Tightly Coupled Components
```
Generator ─→ heavily depends on ─→ Archive (runtime type checks)
         ├──→ Domain (direct coupling)
         └──→ Operators (actually OK - well injected)
```

### After: Plugin-Based with Protocols
```
Core Abstract Framework
├── GeneratorBase (handles metrics, evaluation, common logic)
├── ArchiveProtocol (formal interface, no isinstance checks)
├── DomainProtocol (formal interface, composable)
└── Operator Protocols (already excellent)

Concrete Implementations (Plugins)
├── Generators: EAGenerator, MapElitesGenerator, DEAGenerator, CustomGenerator
├── Archives: GridArchive, CVTArchive, CVTArchiveFixed, YourCustomArchive
└── Domains: BPP, KP, TSP, YourCustomDomain
```

---

## 2. Phase 1: Extract Common Logic

### 2.1 Create BaseGenerator Class

**File**: `digneapy/generators/_base_generator.py`

```python
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple
import numpy as np
from .._core import Instance, Domain, SupportsSolve, P
from .._core._metrics import Logbook, Statistics

class BaseGenerator(ABC):
    """Abstract base class for all Quality-Diversity generators.
    
    Handles:
    - Common evaluation pipeline
    - Descriptor computation
    - Result formatting
    - Statistics tracking
    
    Subclasses implement:
    - generate_population()
    - _generate_offspring()
    - _update_descriptors()
    """
    
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        generations: int = 1000,
        repetitions: int = 1,
        seed: int = 42,
    ):
        self.domain = domain
        self.portfolio = list(portfolio)
        self.generations = generations
        self.repetitions = repetitions
        self.rng = np.random.default_rng(seed)
        self.logbook = Logbook()
    
    def _evaluate_population(
        self,
        population: Sequence[Instance],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Shared evaluation logic (currently duplicated in 2 places).
        
        Returns:
            (performance_scores, diversity_scores)
        """
        performances = np.zeros(len(population), dtype=np.float64)
        # ... shared evaluation code ...
        return performances, diversity_scores
    
    @abstractmethod
    def _generate_population(self) -> np.ndarray:
        """Generate initial population. Override in subclasses."""
        pass
    
    @abstractmethod
    def __call__(self, verbose: bool = False) -> GenResult:
        """Execute the algorithm. Override in subclasses."""
        pass
```

**Migration Path**:
1. Extract `_evaluate_population()` from EAGenerator
2. Extract `_update_descriptors()` into base (with descriptor strategy param)
3. Extract `__compute_fitness()` → `_compute_fitness()` in base

**Benefits**:
- ✅ Eliminates 60% code duplication
- ✅ Single source of truth for evaluation
- ✅ Easier to add new QD algorithms

---

## 3. Phase 2: Define Formal Protocols

### 3.1 ArchiveProtocol (Replace Runtime Type Checks)

**File**: `digneapy/archives/_archive_protocol.py`

```python
from typing import Protocol, Sequence, Optional, Tuple
import numpy as np
from .._core import Instance

class ArchiveProtocol(Protocol):
    """Interface that all archives must implement.
    
    This enables composition over inheritance and removes
    isinstance() type checks from generators.
    """
    
    @property
    def instances(self) -> Sequence[Instance]:
        """Return all stored instances."""
        ...
    
    @property
    def descriptors(self) -> np.ndarray:
        """Return descriptors of all instances (n_instances, n_dims)."""
        ...
    
    @property
    def threshold(self) -> float:
        """Return novelty/fitness threshold."""
        ...
    
    def extend(
        self,
        instances: Sequence[Instance],
        novelty_scores: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None,
    ) -> None:
        """Add instances to archive if novelty_scores >= threshold."""
        ...
    
    def add_to_cell(self, instance: Instance, descriptor: np.ndarray) -> None:
        """Archive-specific: Add instance to cell/region."""
        ...
    
    def index_of(self, descriptors: np.ndarray) -> np.ndarray:
        """Return cell indices for descriptors."""
        ...
    
    def asdict(self) -> dict:
        """Serialize to dictionary."""
        ...
```

**Usage in Generator**:
```python
# BEFORE (type check - fragile)
if not isinstance(archive, (GridArchive, CVTArchive)):
    raise ValueError("...")

# AFTER (protocol - flexible)
def __init__(self, archive: ArchiveProtocol, ...):
    self.archive = archive  # Type checker ensures protocol compliance
```

---

### 3.2 DomainProtocol (Formal Interface)

**File**: `digneapy/_core/domain_protocol.py`

```python
from typing import Protocol, Sequence, Tuple
import numpy as np
from . import Instance, P

class DomainProtocol(Protocol):
    """Interface for problem domains."""
    
    @property
    def name(self) -> str:
        """Domain name (e.g., 'BPP', 'KP', 'TSP')."""
        ...
    
    @property
    def features_size(self) -> int:
        """Number of features for descriptor computation."""
        ...
    
    def generate_instance(self) -> Instance:
        """Generate a random instance in this domain."""
        ...
    
    def mutate(self, instance: Instance) -> Instance:
        """Apply domain-specific mutation."""
        ...
    
    def crossover(self, parent1: Instance, parent2: Instance) -> Instance:
        """Apply domain-specific crossover."""
        ...
    
    def get_solver(self, solver_name: str) -> SupportsSolve[P]:
        """Get a solver by name (used by portfolio)."""
        ...
```

---

## 4. Phase 3: Extensibility Framework

### 4.1 Generator Registry (Factory Pattern)

**File**: `digneapy/generators/_registry.py`

```python
from typing import Type, Dict, Callable
from ._base_generator import BaseGenerator

class GeneratorRegistry:
    """Central registry for generator implementations.
    
    Enables: plugin discovery, dynamic instantiation, extensibility.
    """
    
    _generators: Dict[str, Type[BaseGenerator]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a generator."""
        def decorator(generator_cls: Type[BaseGenerator]):
            cls._generators[name] = generator_cls
            return generator_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseGenerator:
        """Instantiate a registered generator."""
        if name not in cls._generators:
            raise ValueError(f"Generator '{name}' not registered")
        return cls._generators[name](**kwargs)
    
    @classmethod
    def list_generators(cls) -> list[str]:
        """List all available generators."""
        return list(cls._generators.keys())

# Usage
@GeneratorRegistry.register("ea")
class EAGenerator(BaseGenerator):
    ...

@GeneratorRegistry.register("map-elites")
class MapElitesGenerator(BaseGenerator):
    ...

# Create generators dynamically
gen = GeneratorRegistry.create("ea", domain=my_domain, portfolio=solvers)
```

### 4.2 Archive Registry

**File**: `digneapy/archives/_registry.py`

```python
class ArchiveRegistry:
    """Registry for archive implementations."""
    
    _archives: Dict[str, Type[ArchiveProtocol]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(archive_cls):
            cls._archives[name] = archive_cls
            return archive_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> ArchiveProtocol:
        if name not in cls._archives:
            raise ValueError(f"Archive '{name}' not registered")
        return cls._archives[name](**kwargs)

# Usage
@ArchiveRegistry.register("grid")
class GridArchive(Archive):
    ...

@ArchiveRegistry.register("cvt")
class CVTArchive(GridArchive):
    ...
```

### 4.3 Domain Registry

**File**: `digneapy/domains/_registry.py`

```python
class DomainRegistry:
    """Registry for domain implementations."""
    
    _domains: Dict[str, Type[DomainProtocol]] = {}
    
    @classmethod
    def register(name: str):
        def decorator(domain_cls):
            cls._domains[name] = domain_cls
            return domain_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> DomainProtocol:
        if name not in cls._domains:
            raise ValueError(f"Domain '{name}' not registered")
        return cls._domains[name](**kwargs)

# Usage
@DomainRegistry.register("bpp")
class BinPackingDomain(Domain):
    ...

@DomainRegistry.register("kp")
class KnapsackDomain(Domain):
    ...
```

---

## 5. Phase 4: Configuration System

### 5.1 Generator Configuration

**File**: `digneapy/_core/config.py`

```python
from dataclasses import dataclass
from typing import Optional, Type, Dict, Any
from .generators import BaseGenerator
from .archives import ArchiveProtocol
from .domains import DomainProtocol

@dataclass
class GeneratorConfig:
    """Configuration for generator instantiation."""
    
    generator_type: str  # "ea", "map-elites", "dea"
    domain_type: str     # "bpp", "kp", "tsp"
    archive_type: str    # "grid", "cvt"
    
    # Generator params
    generations: int = 1000
    population_size: int = 100
    
    # Archive params
    archive_params: Dict[str, Any] = None
    
    # Domain params
    domain_params: Dict[str, Any] = None
    
    def create_generator(self) -> BaseGenerator:
        """Factory method to create configured generator."""
        domain = DomainRegistry.create(self.domain_type, **(self.domain_params or {}))
        archive = ArchiveRegistry.create(self.archive_type, **(self.archive_params or {}))
        portfolio = domain.get_default_portfolio()
        
        return GeneratorRegistry.create(
            self.generator_type,
            domain=domain,
            archive=archive,
            portfolio=portfolio,
            generations=self.generations,
            population_size=self.population_size,
        )

# Usage
config = GeneratorConfig(
    generator_type="map-elites",
    domain_type="kp",
    archive_type="cvt",
    generations=100,
    archive_params={"k": 100, "ranges": [(-1, 1), (-1, 1)]},
)
gen = config.create_generator()
results = gen()
```

---

## 6. Migration Guide: Step by Step

### Step 1: Create Base Generator Class
```bash
1. Create: digneapy/generators/_base_generator.py
2. Move _evaluate_population() from EAGenerator to BaseGenerator
3. Move _update_descriptors() to BaseGenerator
4. Update EAGenerator to inherit from BaseGenerator
5. Update MapElitesGenerator to inherit from BaseGenerator
```

### Step 2: Define Protocols
```bash
1. Create: digneapy/archives/_archive_protocol.py
2. Create: digneapy/_core/domain_protocol.py
3. Update existing Archives to satisfy protocol
4. Update existing Domains to satisfy protocol
```

### Step 3: Implement Registries
```bash
1. Create: digneapy/generators/_registry.py
2. Create: digneapy/archives/_registry.py
3. Create: digneapy/domains/_registry.py
4. Register existing generators/archives/domains
5. Update __init__.py to auto-register
```

### Step 4: Add Configuration System
```bash
1. Create: digneapy/_core/config.py
2. Write config tests
3. Update examples to use config system
4. Add YAML config file support (optional)
```

### Step 5: Update Examples
```bash
1. Update all examples/ scripts to use registries/config
2. Add examples/custom_generator.py
3. Add examples/custom_archive.py
4. Add examples/custom_domain.py
```

---

## 7. Adding New Components

### Adding a New Generator

```python
# File: digneapy/generators/cmaes_generator.py
from ._base_generator import BaseGenerator
from ._registry import GeneratorRegistry

@GeneratorRegistry.register("cmaes")
class CMAESGenerator(BaseGenerator):
    """CMA-ES based QD generator."""
    
    def __init__(
        self,
        domain,
        portfolio,
        generations=1000,
        population_size=50,
        sigma=0.3,
        **kwargs
    ):
        super().__init__(domain, portfolio, generations, **kwargs)
        self.population_size = population_size
        self.sigma = sigma
    
    def __call__(self, verbose=False) -> GenResult:
        # Implementation using inherited _evaluate_population()
        # and _update_descriptors()
        pass
```

### Adding a New Archive

```python
# File: digneapy/archives/kdtree_archive.py
from ._archive_protocol import ArchiveProtocol
from ._registry import ArchiveRegistry

@ArchiveRegistry.register("kdtree")
class KDTreeArchive:
    """Archive using KD-tree for fast nearest neighbor queries."""
    
    def __init__(self, ranges, threshold=0.0):
        self._instances = []
        self._descriptors = []
        self._threshold = threshold
    
    @property
    def instances(self):
        return self._instances
    
    @property
    def descriptors(self):
        return np.array(self._descriptors)
    
    def extend(self, instances, novelty_scores=None, descriptors=None):
        # KD-tree specific logic
        pass
    
    def index_of(self, descriptors):
        # Fast kNN lookup
        pass
```

### Adding a New Domain

```python
# File: digneapy/domains/vehicle_routing.py
from ._domain_protocol import DomainProtocol
from ._registry import DomainRegistry

@DomainRegistry.register("vrp")
class VehicleRoutingDomain:
    """Vehicle Routing Problem domain."""
    
    name = "VRP"
    features_size = 10
    
    def generate_instance(self) -> Instance:
        pass
    
    def mutate(self, instance: Instance) -> Instance:
        pass
    
    def crossover(self, parent1: Instance, parent2: Instance) -> Instance:
        pass
    
    def get_solver(self, solver_name: str):
        pass
```

---

## 8. Benefits Summary

| Aspect | Current | After Refactoring |
|--------|---------|-------------------|
| Adding new generator | Rewrite from scratch | Inherit BaseGenerator, ~100 LOC |
| Adding new archive | Tight coupling risk | Implement protocol, standalone |
| Code duplication | 60% in generators | 0%, shared in BaseGenerator |
| Type safety | Runtime checks | Protocol-based, IDE support |
| Extensibility | Hard | Easy (registries + protocols) |
| Plugin discovery | Manual | Automatic (registries) |
| Configuration | Hard-coded | Declarative (YAML/dict based) |
| Testing | Tightly coupled | Mockable protocols |

---

## 9. Backward Compatibility

Keep current API working:

```python
# Old API continues to work
from digneapy import EAGenerator, GridArchive, BinPackingDomain

gen = EAGenerator(
    domain=BinPackingDomain(),
    portfolio=[solver1, solver2],
)
```

Add new API:

```python
# New declarative API
from digneapy import GeneratorConfig

config = GeneratorConfig(
    generator_type="ea",
    domain_type="bpp",
    archive_type="grid",
)
gen = config.create_generator()
```

---

## 10. Testing Strategy

### Protocol Compliance Tests

```python
# tests/test_archive_protocol.py
def test_archive_implements_protocol():
    """Verify archives satisfy ArchiveProtocol."""
    archive = GridArchive(...)
    
    # Check all protocol methods exist
    assert hasattr(archive, 'instances')
    assert hasattr(archive, 'descriptors')
    assert hasattr(archive, 'extend')
    assert hasattr(archive, 'index_of')
```

### Registry Tests

```python
# tests/test_registries.py
def test_generator_registry():
    """Verify generators register correctly."""
    assert "ea" in GeneratorRegistry.list_generators()
    
    gen = GeneratorRegistry.create("ea", ...)
    assert isinstance(gen, BaseGenerator)
```

---

## Timeline & Effort Estimate

| Phase | Effort | Duration |
|-------|--------|----------|
| Phase 1: BaseGenerator | Medium | 1-2 days |
| Phase 2: Protocols | Low | 1 day |
| Phase 3: Registries | Low | 1 day |
| Phase 4: Config | Medium | 1-2 days |
| Phase 5: Examples | Low | 1 day |
| **Total** | **~1 week** | |

---

## Next Steps

1. Start with Phase 1 (extract BaseGenerator)
2. Create comprehensive tests for each phase
3. Update documentation with examples
4. Add plugin templates in `docs/plugin_templates/`
5. Publish `CONTRIBUTING.md` with extension guidelines
