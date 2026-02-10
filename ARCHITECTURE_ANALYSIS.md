# DIGNEApy Architecture Analysis

## Executive Summary
DIGNEApy is a quality-diversity optimization framework with **moderate coupling** and **standardized interfaces** using Python Protocols. The codebase has clear abstractions but exhibits **tight coupling in generators to specific archive types** and **domain-specific logic diffusion**.

---

## 1. GENERATORS

### 1.1 Generator Protocol/Interface
```python
# generators.py, lines 67-73
class Generator(Protocol):
    """Protocol to type check all generators of instances types in digneapy"""

    def __call__(self, *args, **kwargs) -> GenResult: ...

    def _update_descriptors(
        self,
        population: np.ndarray,
        portfolio_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]: ...
```

**Key Observations:**
- Minimal protocol - only 2 required methods
- `GenResult` is the standard output container
- Allows flexibility in implementation details

### 1.2 Three Generator Implementations

#### A. **EAGenerator** (Evolutionary Algorithm with Novelty Search)
```python
# generators.py, lines 75-387
class EAGenerator(Generator, RNG):
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        novelty_approach: NS,  # ← Tightly coupled to NS algorithm
        pop_size: int = 100,
        generations: int = 1000,
        solution_set: Optional[Archive] = None,
        descriptor_strategy: str = "features",
        transformer: Optional[SupportsTransform] = None,
        # ... operators
        crossover: Crossover = uniform_crossover,
        mutation: Mutation = uniform_one_mutation,
        selection: Selection = binary_tournament_selection,
        replacement: Replacement = generational_replacement,
        performance_function: PerformanceFn = max_gap_target,
        phi: float = 0.85,
        seed: int = 42,
    ):
```

**Dependencies:**
- `Domain` - for instance generation & feature extraction
- `Portfolio` (SupportsSolve) - solver evaluation
- `NS` (Novelty Search) - diversity guidance
- `Archive` - optional solution storage
- `Operators` - crossover, mutation, selection, replacement
- `SupportsTransform` - optional descriptor transformation
- `PerformanceFn` - performance scoring

**Key Methods:**
1. `__call__()` - Main loop:
   - Initial population generation
   - Fitness computation (blended performance + novelty)
   - Offspring generation via operators
   - Archive updates with feasible solutions
2. `_evaluate_population()` - Runs all solvers in portfolio on all instances
3. `_update_descriptors()` - Applies descriptor strategy (features/performance/custom)
4. `__reproduce()` - Creates offspring via crossover + mutation

**Tightness Metrics:**
- ✅ Decoupled operators (injected as callables)
- ⚠️ Tight coupling to `NS` class (hardcoded novelty_approach parameter)
- ⚠️ Tightly coupled fitness composition (phi-blended performance+novelty)
- ✅ Descriptor strategy is pluggable

**Code Snippet - Fitness Calculation:**
```python
def __compute_fitness(
    self, performance_biases: np.ndarray, novelty_scores: np.ndarray
) -> np.ndarray:
    phi_r = 1.0 - self.phi
    fitness = np.zeros(len(performance_biases))
    fitness = (fitness * self.phi) + (novelty_scores * phi_r)  # ← Hardcoded blending
    return fitness
```

#### B. **MapElitesGenerator** (Quality-Diversity via MAP-Elites)
```python
# generators.py, lines 390-520
class MapElitesGenerator(Generator, RNG):
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        initial_pop_size: int,
        generations: int,
        archive: GridArchive | CVTArchive,  # ← Tightly typed archive
        mutation: Mutation,
        repetitions: int,
        descriptor: str,  # ← String-based descriptor lookup
        performance_function: PerformanceFn = max_gap_target,
        seed: int = 42,
    ):
        if not isinstance(archive, (GridArchive, CVTArchive)):  # ← Runtime type check!
            raise ValueError(...)
```

**Key Differences from EAGenerator:**
- NO novelty search object - uses native archive divisions as solution space
- **Tightly typed archive** - only accepts GridArchive or CVTArchive
- Descriptor strategy resolved via string lookup in `DESCRIPTORS` dictionary
- Batch mutation via `batch_uniform_one_mutation`
- Parent selection from archive cells (stored individuals)

**Tightness Issues:**
- 🔴 **Runtime type checking** for archive (should use Protocol)
- 🔴 **Tight coupling** to specific archive implementations
- ⚠️ String-based descriptor lookup (fragile, no IDE support)

#### C. **DEAGenerator** (Dominated Evolutionary Algorithm)
```python
# generators.py, lines 523-664
class DEAGenerator(EAGenerator):
    """Subclass of EAGenerator - uses Dominated Novelty Search instead of novelty_approach"""
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        pop_size: int = 128,
        offspring_size: int = 128,
        generations: int = 1000,
        k: int = 15,  # ← k-neighbors parameter for DNS
        descriptor_strategy: str = "features",
        # ... others
    ):
```

**Key Implementation:**
```python
def __call__(self, verbose: bool = False) -> GenResult:
    # ... init & evaluation ...
    
    (
        sorted_descriptors,
        sorted_performances,
        sorted_competition_fitness,
        sorted_indexing,
    ) = dominated_novelty_search(  # ← Calls DNS function directly
        descriptors=combined_descriptors,
        performances=combined_performances,
        k=self.k,
        force_feasible_only=True,
    )
```

**Coupling:**
- ✅ Leverages parent EAGenerator's infrastructure
- ✅ Uses functional DNS algorithm (not class-based)
- ⚠️ Overrides parent's entire `__call__()` method

---

## 2. ARCHIVES

### 2.1 Archive Base Class
```python
# archives/_base_archive.py, lines 1-160
class Archive:
    """Stores a collection of diverse Instances"""
    
    def __init__(
        self,
        threshold: float,
        instances: Optional[Sequence[Instance]] = None,
        dtype=np.float64,
    ):
        self._storage = {"instances": [], "descriptors": []}
        self._threshold = threshold
    
    def extend(
        self,
        instances: Sequence[Instance],
        novelty_scores: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None,
    ):
        """Extends archive with feasible instances"""
```

**Interface Design:**
- Simple unstructured storage: `{instances, descriptors}`
- Threshold-based filtering (configurable)
- Limited public API: `extend()`, iteration, indexing

**Pain Points:**
- No abstract methods - Archive is NOT abstract! 
- "Protocol" only exists implicitly through usage
- No formal `add()`, `query()`, or `purge()` contract

### 2.2 GridArchive (Structured via Regular Grid)
```python
# archives/_grid_archive.py, lines 1-200
class GridArchive(Archive):
    def __init__(
        self,
        dimensions: Sequence[int],  # cells per dimension
        ranges: Sequence[Tuple[float, float]],  # measure space bounds
        instances: Optional[Iterable[Instance]] = None,
        eps: float = 1e-6,
    ):
        self._dimensions = np.asarray(dimensions)
        self._grid: Dict[int, np.ndarray] = {}  # cell_id -> instance
        self._storage: Dict[int, Instance] = {}  # single elite per cell
        self._boundaries = ...  # cell boundaries computed from ranges
```

**Key Properties:**
- **One elite per cell** - best solution per grid cell
- **Continuous measure space** → discrete cells via binning
- Index computation: `flat_index = tuple(np.digitize(descriptors, boundaries))`

**Implementation:**
- Uses linspace to create cell boundaries
- Stores instances in Dictionary keyed by cell index
- `coverage` property: `len(filled_cells) / total_cells`

### 2.3 CVTArchive (Continuity Voronoi Tessellation)
```python
# archives/_cvt_archive.py, lines 1-200
class CVTArchive(GridArchive, RNG):
    """Extends GridArchive with Centroidal Voronoi Tessellation"""
    
    def __init__(
        self,
        k: int,  # number of centroids/regions
        ranges: Sequence[Tuple[float, float]],
        n_samples: int,  # bootstrap sampling
        centroids: Optional[npt.NDArray | str] = None,
        samples: Optional[npt.NDArray | str] = None,
    ):
        # Inherits GridArchive structure but with k regions
        self._kmeans = KMeans(n_clusters=self._k, n_init=1)
        self._kmeans.fit(self._samples)
        self._centroids = self._kmeans.cluster_centers_
        self._kdtree = KDTree(self._centroids, metric="euclidean")
```

**Key Differences:**
- **Homogeneous Geometric Regions** - k-means clustering instead of uniform grid
- Uses KDTree for nearest centroid lookup
- Can load/save pre-computed centroids & samples

**Coupling Issue:**
- 🔴 Inherits from GridArchive but redefines core behavior
- Awkward inheritance hierarchy - CVTArchive IS-NOT-A GridArchive semantically

---

## 3. DOMAINS

### 3.1 Domain Base Class
```python
# _core/_domain.py, lines 1-150
class Domain(ABC, RNG):
    """Defines problem domain"""
    
    def __init__(
        self,
        dimension: int,
        bounds: Sequence[tuple],
        dtype=np.float64,
        name: str = "Domain",
        feat_names: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        self._dimension = dimension
        self._bounds = bounds  # [(lb, ub), ...]
        self._lbs = np.array(ranges[0], dtype=dtype)  # lower bounds vector
        self._ubs = np.array(ranges[1], dtype=dtype)  # upper bounds vector
    
    @abstractmethod
    def generate_instances(self, n: int = 1) -> List[Instance]:
        """Generates N instances"""
        raise NotImplementedError()
    
    @abstractmethod
    def generate_problems_from_instances(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[Problem]:
        """Converts instances to Problem objects for solving"""
        raise NotImplementedError()
    
    @abstractmethod
    def extract_features(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> np.ndarray:
        """Extracts descriptor features from instances"""
        raise NotImplementedError()
    
    @abstractmethod
    def extract_features_as_dict(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[Dict[str, np.float32]]:
        """Returns features as dictionaries"""
        raise NotImplementedError()
```

**Core Responsibilities:**
1. Instance generation (random or from parameters)
2. Problem creation (instances → solvable problems)
3. Feature extraction (instances → descriptors)

### 3.2 Domain Implementations

#### **BPP Domain** (Bin Packing Problem)
```python
# domains/bpp.py
class BPP(Problem):
    def __init__(self, items: Iterable[int], capacity: int, seed: int = 42):
        self._items = tuple(items)
        self._capacity = capacity
        bounds = list((0, dim - 1) for _ in range(dim))  # bin indices
        super().__init__(dimension=dim, bounds=bounds, name="BPP", seed=seed)
    
    def evaluate(self, individual: Sequence | Solution) -> tuple[float]:
        """Falkenauer fitness: sum of (fill_k/C)^2 / num_bins"""
        used_bins = np.max(individual).astype(int) + 1
        fill_i = np.zeros(used_bins)
        for item_idx, bin in enumerate(individual):
            fill_i[bin] += self._items[item_idx]
        fitness = sum((f_i / self._capacity) ** 2 for f_i in fill_i) / used_bins
        return (fitness,)
```

**Instance Representation:** `[bin_0, bin_1, ..., bin_n]` where bin_i ∈ [0, num_bins)

#### **KP Domain** (0/1 Knapsack)
```python
# domains/kp.py
class Knapsack(Problem):
    def __init__(
        self,
        profits: Sequence[int],
        weights: Sequence[int],
        capacity: int = 0,
    ):
        self.weights = weights
        self.profits = profits
        self.capacity = capacity
        self.penalty_factor = 100.0  # constraint violation penalty
    
    def evaluate(self, individual: Sequence | Solution | np.ndarray) -> Tuple[float]:
        profit = np.dot(individual, self.profits)
        packed = np.dot(individual, self.weights)
        penalty = max(0, packed - capacity) * penalty_factor
        return (profit - penalty,)

class KnapsackDomain(Domain):
    """Generates random KP instances"""
    def __init__(
        self,
        dimension: int = 50,
        min_p: int = 1,
        min_w: int = 1,
        max_p: int = 1_000,
        max_w: int = 1_000,
        capacity_approach: str = "evolved",  # "evolved", "percentage", "fixed"
        max_capacity: int = int(1e5),
        capacity_ratio: float = 0.8,
    ):
        # bounds = [(1.0, max_capacity)] + [(min_w, max_w), (min_p, max_p), ...]
        #          for 2*dimension items mixed weights/profits
```

**Instance Representation:** `[capacity, w_0, p_0, w_1, p_1, ..., w_n, p_n]`

#### **TSP Domain** (Travelling Salesman Problem)
- Not fully shown in readings but follows similar pattern
- Instance: array of city coordinates or distance matrix

**Domain Coupling Issues:**
- ⚠️ Domains are tightly specific to their problem
- ⚠️ No way to change how instances are generated without subclassing
- ⚠️ Feature extraction is hardcoded per domain
- ✅ Feature names documented in `feat_names` attribute

---

## 4. OPERATORS

### 4.1 Operator Protocols (Type Aliases)

```python
# operators/__init__.py

# Crossover: (Individual, Individual) -> Individual
Crossover = Callable[[IndType, IndType], IndType]

# Mutation: (Individual, Bounds) -> Individual
Mutation = Callable[[IndType, Sequence[Tuple]], IndType]

# Selection: (Population) -> Individual
Selection = Callable[[Sequence[IndType]], IndType]

# Replacement: (Population, Offspring) -> Population
Replacement = Callable[[Sequence[IndType], Sequence[IndType]], Sequence[IndType]]
```

**Design Pattern:** Protocol-via-TypeAlias (not formal Protocol classes)

### 4.2 Implementations

#### **Crossover Operators**
```python
# operators/_crossover.py

def uniform_crossover(
    individual: IndType, other: IndType, cxpb: np.float64 = 0.5
) -> IndType:
    """Swaps genes with probability cxpb"""
    probs = np.random.default_rng().random(size=len(individual))
    genotype = np.where(probs <= cxpb, individual, other)
    return individual.clone_with(variables=genotype)

def one_point_crossover(individual: IndType, other: IndType) -> IndType:
    """Single split point"""
    offspring = individual.clone()
    cross_point = np.random.integers(0, len(individual))
    offspring[cross_point:] = other[cross_point:]
    return offspring
```

#### **Mutation Operators**
```python
# operators/_mutation.py

def uniform_one_mutation(individual: IndType, bounds: Sequence[Tuple]) -> IndType:
    """Mutates one random gene uniformly in [lb, ub]"""
    mutation_point = rng.integers(0, len(individual))
    new_value = rng.uniform(bounds[mutation_point][0], bounds[mutation_point][1])
    individual[mutation_point] = new_value
    return individual

def batch_uniform_one_mutation(
    population: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> np.ndarray:
    """Vectorized mutation for entire population"""
    mutation_points = rng.integers(0, dimension, size=len(population))
    new_values = rng.uniform(lb[mutation_points], ub[mutation_points])
    population[np.arange(len(population)), mutation_points] = new_values
    return population
```

#### **Selection Operator**
```python
def binary_tournament_selection(population: Sequence[IndType]) -> IndType:
    """Selects best of two random individuals"""
    idx1, idx2 = np.random.integers(0, len(population), size=2)
    return max(population[idx1], population[idx2], key=attrgetter("fitness"))
```

#### **Replacement Operators**
```python
def generational_replacement(
    current_population: Sequence[IndType],
    offspring: Sequence[IndType],
) -> Sequence[IndType]:
    """Offspring replace entire population"""
    return offspring[:]

def first_improve_replacement(
    current_population: Sequence[IndType],
    offspring: Sequence[IndType],
) -> Sequence[IndType]:
    """Element-wise best of current vs offspring"""
    return [max(a, b, key=attrgetter("fitness")) 
            for a, b in zip(current_population, offspring)]

def elitist_replacement(
    current_population: Sequence[IndType],
    offspring: Sequence[IndType],
    hof: int = 1,  # Hall of Fame size
) -> list[IndType]:
    """Keep top hof from combined, fill rest from offspring"""
    combined = current_population + offspring
    top_hof = nsmallest(hof, combined, key=attrgetter("fitness"))
    remaining = ...
    return top_hof + remaining
```

**Operator Tightness:**
- ✅ Highly decoupled - all are simple callables
- ✅ Pluggable - generators accept them as parameters
- ⚠️ No formal Protocol class (just type aliases)
- ⚠️ Operators assume Instance/Solution have `.fitness` attribute
- ⚠️ Bounds passing required by Mutation - couples to Domain bounds structure

---

## 5. OVERALL ARCHITECTURE

### 5.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    GENERATOR (__call__)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INIT:                                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Domain.generate_instances()  ──→  Population         │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ EVALUATION LOOP:                                     │  │
│  │ - Domain.generate_problems_from_instances()          │  │
│  │ - Portfolio[*].solve(Problem)  ──→  scores           │  │
│  │ - PerformanceFn(scores)  ──→  performance_bias       │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ DESCRIPTOR EXTRACTION:                               │  │
│  │ - Domain.extract_features()  OR                       │  │
│  │ - Descriptor lookup  OR                              │  │
│  │ - Transformer.transform()                            │  │
│  │         ──→  descriptors                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ↓                                    │
│  GENERATION LOOP (for each generation):                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ VARIATIONAL:                                         │  │
│  │ - Selection(population)  ──→  parent1, parent2       │  │
│  │ - Crossover(parent1, parent2)  ──→  offspring        │  │
│  │ - Mutation(offspring, bounds)  ──→  offspring        │  │
│  │         ──→  New Population                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                    │
│         ┌───────────────┴────────────────┐                  │
│         ↓                               ↓                   │
│    [EAGenerator]                  [MapElites]              │
│    ┌──────────────────┐          ┌──────────────────┐      │
│    │ NS Algorithm:    │          │ Archive Storage: │      │
│    │ Novelty.score()  │          │ GridArchive OR   │      │
│    │ Archive.extend() │          │ CVTArchive       │      │
│    │                  │          │ .extend()        │      │
│    └──────────────────┘          └──────────────────┘      │
│         │                                                   │
│         ↓                                                   │
│    ┌──────────────────────────────────────────────┐        │
│    │ FITNESS COMPUTATION:                         │        │
│    │ - EAGenerator: phi * perf + (1-phi) * novelty│        │
│    │ - MapElites: fitness = perf ONLY             │        │
│    │ - DEAGenerator: DNS(combined_descriptors)    │        │
│    └──────────────────────────────────────────────┘        │
│         │                                                   │
│         ↓                                                   │
│    Replacement(population, offspring)                      │
│         │                                                   │
│         └─→ NEXT GENERATION                                │
│                                                             │
│  OUTPUT:  GenResult(instances=archive, history=logbook)    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Component Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│                  GENERATORS                             │
├─────────────────────────────────────────────────────────┤
│  EAGenerator │ MapElitesGenerator │ DEAGenerator        │
└────────┬─────────┬────────────────┬────────────┬────────┘
         │         │                │            │
    ┌────┴─┬───────┴────┬───────────┴─┐         │
    │      │            │             │         │
    ↓      ↓            ↓             ↓         ↓
┌────────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
│   DOMAIN       │ │ ARCHIVE  │ │ NS      │ │OPERATORS │
├────────────────┤ ├──────────┤ ├─────────┤ ├──────────┤
│ generate_inst()│ │ Archive  │ │ NS()    │ │Crossover │
│ gen_problems() │ │├GridArch │ │ DNS()   │ │Mutation  │
│extract_features│ │└CVTArch  │ │         │ │Selection │
│DomainSpecific: │ │          │ │         │ │Replacem. │
│ - BPP          │ │          │ │         │ │          │
│ - KA           │ │          │ │         │ │          │
│ - TSP          │ │          │ │         │ │          │
└────────────────┘ └──────────┘ └─────────┘ └──────────┘
         │              │            │              │
         │ (Problem)    │            │              │
         └─────────┬────┘            │              │
                   │                 │              │
   ┌───────────────┴─────────────────┴──────────────┘
   │
   ↓
┌──────────────┐
│  INSTANCE    │
├──────────────┤
│ variables[]  │
│ fitness      │
│ descriptor   │
│ performance  │
│ novelty      │
│ features     │
│ port_scores  │
└──────────────┘
```

### 5.3 Coupling Analysis Matrix

| Component | Domain | Archive | Operators | Transformers | NS | Portfolio |
|-----------|--------|---------|-----------|--------------|-----|-----------|
| **EAGenerator** | 🔴 Tight | ✅ Pluggable | ✅ Injected | ✅ Optional | 🔴 Hard | ✅ Injected |
| **MapElites** | 🔴 Tight | 🔴 Typed | N/A | N/A | N/A | ✅ Injected |
| **DEAGenerator** | 🔴 Tight | ✅ Implicit | ✅ Inherited | ✅ Inherited | ✅ Implicit | ✅ Injected |
| **Archive** | N/A | - | N/A | N/A | N/A | N/A |
| **GridArchive** | N/A | - | N/A | N/A | N/A | N/A |
| **CVTArchive** | N/A | 🔴 Inherits GridArch | N/A | N/A | N/A | N/A |
| **Domain** | - | N/A | N/A | N/A | N/A | ✅ Provided |

**Legend:** 🔴 Tightly Coupled | ⚠️ Moderately Coupled | ✅ Loosely Coupled

---

## 6. EXTENSIBILITY PAIN POINTS

### 6.1 Adding a New Generator Type

**Current Process:**
1. Create new class inheriting from `Generator` Protocol
2. Implement `__call__()` and `_update_descriptors()`
3. Must duplicate significant logic:
   - Population evaluation loop (same in all 3 generators)
   - Descriptor extraction (duplicated logic)
   - Instance creation from arrays

**Problem Code (Duplicated):**
```python
# In ALL generators:
def _evaluate_population(self, population: Sequence[Instance]) -> Tuple[np.ndarray, np.ndarray]:
    solvers_scores = np.zeros(shape=(len(population), len(self.portfolio), self.repetitions))
    problems_to_solve = self.domain.generate_problems_from_instances(population)
    for j, problem in enumerate(problems_to_solve):
        for i, solver in enumerate(self.portfolio):
            scores = np.zeros(self.repetitions)
            for rep in range(self.repetitions):
                scores[rep] = max(solver(problem), key=attrgetter("fitness")).fitness
            solvers_scores[j, i, :] = scores
    # ... same evaluation pattern in EAGenerator, MapElitesGenerator, DEAGenerator
```

**Solution:** Extract common evaluation into base class or utility function.

### 6.2 Adding a New Archive Type

**Current Issues:**
1. **Type Checking:** MapElitesGenerator uses runtime `isinstance()` check
   ```python
   if not isinstance(archive, (GridArchive, CVTArchive)):
       raise ValueError(...)
   ```
   → Cannot accept compatible archive types without modifying generator

2. **Awkward Inheritance:** CVTArchive inherits from GridArchive despite different semantics
   ```python
   class CVTArchive(GridArchive, RNG):  # Wrong! CVT is NOT-A GridArchive
   ```

3. **Missing Protocol:** No formal `ArchiveProtocol` to express contract

**Solution:** Define Archive Protocol, use composition or proper inheritance

```python
# Proposed:
class ArchiveProtocol(Protocol):
    @property
    def instances(self) -> Sequence[Instance]: ...
    
    @property
    def descriptors(self) -> np.ndarray: ...
    
    def extend(self, instances, descriptors, novelty_scores=None): ...
    
    def __len__(self) -> int: ...
    
    def __getitem__(self, key): ...
```

### 6.3 Adding a New Domain

**Current Process:**
1. Subclass `Domain` (ABC)
2. Implement 4 abstract methods:
   - `generate_instances(n)` 
   - `generate_problems_from_instances()`
   - `extract_features()`
   - `extract_features_as_dict()`

3. Define problem-specific objects (BPP, Knapsack, etc.)

**Pain Point - Tight Domain Logic:**
```python
# Every domain must implement feature extraction:
class KnapsackDomain(Domain):
    def extract_features(self, instances) -> np.ndarray:
        # Domain-specific feature extraction
        # No reuse of logic across domains
        ...
```

**Problem:** No shared infrastructure for:
- Instance validation
- Feasibility checking
- Feature caching
- Feature normalization

### 6.4 Adding New Operators

**✅ Good News:** Operators are HIGHLY extensible!

```python
# Easy to add new crossover:
def my_crossover(ind1: IndType, ind2: IndType) -> IndType:
    # Custom implementation
    return modified_offspring

# Use in generator:
gen = EAGenerator(..., crossover=my_crossover)
```

**Minor Issue:** Operators assume `.clone()` and `.clone_with()` on individuals
- Limits compatibility to Instance/Solution types only

### 6.5 Adding New Descriptor Strategies

**Current Mechanism - String-based lookup:**
```python
# generators.py, line 145
try:
    self._desc_key = descriptor_strategy
    self._descriptor_strategy = DESCRIPTORS[self._desc_key]
except KeyError:
    print(f"Descriptor: {descriptor_strategy} not available. Using features.")
    self._descriptor_strategy = DESCRIPTORS["instance"]
```

**Issues:**
- 🔴 String-based (no IDE support, fragile)
- 🔴 Silent fallback on missing key
- ⚠️ Tight coupling to global `DESCRIPTORS` dict

**Unknown:** Where is DESCRIPTORS defined?
- Likely in `descriptors.py` (not fully examined)

---

## 7. DESIGN PATTERNS OBSERVED

### 7.1 Protocol-Based Abstraction (Good)
```python
class Generator(Protocol):
    def __call__(self, *args, **kwargs) -> GenResult: ...
    def _update_descriptors(...) -> Tuple: ...
```
✅ Enables structural subtyping, duck typing friendly

### 7.2 Mix-in Pattern (RNG)
```python
class Domain(ABC, RNG):
    def __init__(self, seed: Optional[int] = None):
        self.initialize_rng(seed=seed)

class CVTArchive(GridArchive, RNG):
    ...
```
✅ Adds random number generation capability

### 7.3 String-based Configuration (Fragile)
```python
descriptor_strategy: str = "features"
self._descriptor_strategy = DESCRIPTORS[self._desc_key]  # Dict lookup
```
❌ No compile-time checking, silent failures

### 7.4 Type Aliases for Protocols (Implicit)
```python
Crossover = Callable[[IndType, IndType], IndType]
Mutation = Callable[[IndType, Sequence[Tuple]], IndType]
```
✅ Clear intent, lightweight
⚠️ Should be formal Protocol classes for better IDE support

### 7.5 Dataclass Result Container
```python
@dataclass
class GenResult:
    target: str
    instances: np.ndarray
    history: Logbook
    metrics: Optional[pd.Series] = None
```
✅ Clean, immutable return type

### 7.6 Strategy Pattern with Default Functions
```python
def __init__(
    self,
    crossover: Crossover = uniform_crossover,
    mutation: Mutation = uniform_one_mutation,
    ...
):
```
✅ Strategies injected, defaults provided

---

## 8. TESTING AND VERIFICATION RECOMMENDATIONS

### Code Smells Identified:

1. **Duplicate Code:**
   - `_evaluate_population()` identical in EAGenerator & MapElitesGenerator
   - `_update_descriptors()` duplicated logic across generators

2. **Magic Numbers:**
   - `phi = 0.85` - hard-coded default
   - `self.penalty_factor = 100.0` in Knapsack (constraint penalty)
   - `RNG seed = 42` throughout

3. **Silent Failures:**
   - Descriptor strategy falls back silently on missing key
   - Archive type checking at runtime instead of compile-time

4. **Tight Coupling:**
   - MapElitesGenerator tightly typed to GridArchive | CVTArchive
   - EAGenerator requires NS object
   - DEAGenerator overrides entire parent `__call__()` method

5. **Missing Abstractions:**
   - No ArchiveProtocol (implicit only)
   - No formal OperatorProtocol (just type aliases)
   - No BaseGenerator superclass to factor common logic

---

## 9. REFACTORING RECOMMENDATIONS

### 9.1 Extract Common Generator Logic
```python
# Before: Duplicated code across all generators
# After:
class BaseGenerator(Generator, RNG):
    def _evaluate_population(self, population):
        # Shared evaluation logic
        ...
    
    def _extract_descriptors(self, population, portfolio_scores):
        # Shared descriptor extraction
        ...

class EAGenerator(BaseGenerator):
    def __call__(self, verbose=False):
        # Only EA-specific logic here
        ...
```

### 9.2 Define Archive Protocol
```python
class ArchiveProtocol(Protocol):
    @property
    def instances(self) -> Sequence[Instance]: ...
    
    @property
    def descriptors(self) -> np.ndarray: ...
    
    @property
    def n_solutions(self) -> int: ...
    
    def extend(
        self,
        instances: Sequence[Instance],
        descriptors: np.ndarray,
        novelty_scores: Optional[np.ndarray] = None
    ) -> None: ...
    
    def __len__(self) -> int: ...
    
    def __iter__(self) -> Iterator[Instance]: ...

# Then use:
class MapElitesGenerator(Generator, RNG):
    def __init__(
        self,
        archive: ArchiveProtocol,  # Protocol instead of Union type
        ...
    ):
        ...
```

### 9.3 Configuration Objects Instead of Strings
```python
# Before:
descriptor_strategy: str = "features"

# After:
from enum import Enum

class DescriptorStrategy(Enum):
    FEATURES = "features"
    PERFORMANCE = "performance"
    CUSTOM = "custom"

class GeneratorConfig:
    descriptor_strategy: DescriptorStrategy = DescriptorStrategy.FEATURES
    descriptor_fn: Optional[SupportsTransform] = None
```

### 9.4 Separate Concerns - Evaluation vs Evolution
```python
# Create evaluator module
class InstanceEvaluator:
    def __init__(self, portfolio: Iterable[SupportsSolve], domain: Domain):
        self.portfolio = portfolio
        self.domain = domain
    
    def evaluate(self, population) -> Tuple[np.ndarray, np.ndarray]:
        # Single source of truth for evaluation logic
        ...
```

---

## 10. METRICS SUMMARY

| Metric | Score | Notes |
|--------|-------|-------|
| **API Clarity** | 8/10 | Protocols well-defined, generators clear |
| **Extensibility** | 6/10 | Operators great, domains/archives harder |
| **Code Reuse** | 5/10 | Significant duplication across generators |
| **Type Safety** | 6/10 | Protocol-based but runtime checks present |
| **Test Coverage** | ? | Unknown without seeing test files |
| **Documentation** | 6/10 | Docstrings present, architecture unclear |
| **Coupling** | 5/10 | Generators tightly bound to domains |
| **Modularity** | 7/10 | Clear component separation |

---

## 11. KEY FINDINGS SUMMARY

### ✅ Strengths
1. Clean Protocol-based generator interface
2. Well-separated domain implementations
3. Highly extensible operator system
4. Standard output format (GenResult)
5. Mix-in pattern for shared functionality (RNG)

### ❌ Weaknesses
1. **Generator duplication** - 3 generators share 60% code
2. **Archive coupling** - MapElites hardcodes archive types
3. **String-based configuration** - Descriptor lookup fragile
4. **Inheritance misuse** - CVTArchive inherits from GridArchive incorrectly
5. **No base generator class** - Repeated patterns force copy-paste

### ⚠️ Moderate Issues
1. Runtime type checking instead of Protocols
2. Missing ArchiveProtocol definition
3. Silent fallbacks on configuration errors
4. Tight domain coupling in EAGenerator

### 🎯 Priority Improvements
1. Extract `BaseGenerator` with shared evaluation logic
2. Define `ArchiveProtocol` to replace runtime checks
3. Refactor `CVTArchive` to composition instead of inheritance
4. Use `Enum` for descriptor strategies instead of strings
5. Add configuration object layer (avoiding Dict lookups)

