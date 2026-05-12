# DIGNEApy Architecture - Answers to Specific Questions

## Question 1: Generators

### What is the Generator protocol/interface?

**Location:** [generators.py](generators.py#L67-L73)

```python
class Generator(Protocol):
    """Protocol to type check all generators of instances types in digneapy"""

    def __call__(self, *args, **kwargs) -> GenResult: ...

    def _update_descriptors(
        self,
        population: np.ndarray,
        portfolio_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]: ...
```

**Minimal contract:**
- `__call__()`: Run generator, return `GenResult(target, instances, history, metrics)`
- `_update_descriptors()`: Transform population evaluations to descriptors for diversity

---

### How do EAGenerator, MapElitesGenerator, and DEAGenerator currently work?

#### **EAGenerator** - Evolutionary Algorithm + Novelty Search

**Key Code:** [generators.py](generators.py#L75-L387)

```python
class EAGenerator(Generator, RNG):
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        novelty_approach: NS,  # ← REQUIRED novelty search object
        pop_size: int = 100,
        generations: int = 1000,
        solution_set: Optional[Archive] = None,
        descriptor_strategy: str = "features",
        transformer: Optional[SupportsTransform] = None,
        # ... operators (crossover, mutation, selection, replacement)
        phi: float = 0.85,  # ← 85% performance, 15% novelty blend
        seed: int = 42,
    ):
```

**Execution Loop:**
```python
def __call__(self, verbose: bool = False) -> GenResult:
    # 1. INITIALIZATION
    population = domain.generate_instances(n=pop_size)
    perf_biases, portfolio_scores = _evaluate_population(population)
    descriptors, features = _update_descriptors(population, portfolio_scores)
    
    # 2. MAIN LOOP (generations times)
    for gen in range(generations):
        # 2a. VARIATION
        offspring = _generate_offspring(pop_size)  # via selection + crossover + mutation
        
        # 2b. EVALUATION
        off_perf, off_scores = _evaluate_population(offspring)
        off_desc, off_feat = _update_descriptors(offspring, off_scores)
        
        # 2c. FITNESS COMPUTATION
        novelty_scores = novelty_approach(off_desc)  # ← k-NN distances
        fitness = (phi * perf_bias) + ((1 - phi) * novelty_scores)
        
        # 2d. ARCHIVE EXTENSION
        novelty_approach.archive.extend(feasible_offspring, descriptors)
        
        # 2e. REPLACEMENT
        population = replacement(population, offspring_with_fitness)
        
        # 2f. LOGGING
        logbook.update(generation, population)
    
    # 3. RETURN best from archive or population
    return GenResult(target, archive_instances, logbook)
```

**Fitness Calculation - The Heart:**
```python
def __compute_fitness(self, perf_bias: np.ndarray, novelty_scores: np.ndarray) -> np.ndarray:
    phi_r = 1.0 - self.phi
    fitness = np.zeros(len(perf_bias))
    fitness = (fitness * self.phi) + (novelty_scores * phi_r)  # Hardcoded blend!
    return fitness
```

**Quality-Diversity Balance:**
- `phi = 0.85` → 85% weight on performance, 15% on novelty (default)
- `phi = 0.5` → Equal weighting
- `phi = 1.0` → Pure performance (no novelty)

---

#### **MapElitesGenerator** - Map-Elites Quality-Diversity

**Key Code:** [generators.py](generators.py#L390-L520)

```python
class MapElitesGenerator(Generator, RNG):
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        initial_pop_size: int,
        generations: int,
        archive: GridArchive | CVTArchive,  # ← TypeUnion enforced!
        mutation: Mutation,  # ← ONLY mutation, no crossover
        repetitions: int,
        descriptor: str,  # ← String descriptor lookup
        performance_function: PerformanceFn = max_gap_target,
        seed: int = 42,
    ):
        if not isinstance(archive, (GridArchive, CVTArchive)):  # 🔴 RUNTIME CHECK!
            raise ValueError(...)
```

**Execution Loop:**
```python
def __call__(self, verbose: bool = False) -> GenResult:
    # 1. INITIALIZATION
    instances = domain.generate_instances(n=initial_pop_size)
    perf_biases, _ = _evaluate_population(instances)
    descriptors, _ = _update_descriptors(instances)
    
    archive.extend(instances, descriptors)  # ← Populate archive
    
    # 2. MAIN LOOP
    for generation in range(generations):
        # 2a. SELECT from archive cells (not external population)
        indices = rng.choice(archive.filled_cells, size=initial_pop_size)
        parents = archive[indices]
        
        # 2b. MUTATE ONLY (no crossover)
        offspring = batch_uniform_one_mutation(parents, domain._lbs, domain._ubs)
        
        # 2c. EVALUATE
        off_perf, off_scores = _evaluate_population(offspring)
        off_desc, _ = _update_descriptors(offspring, off_scores)
        
        # 2d. ADD TO ARCHIVE (best per cell survives)
        archive.extend(offspring, off_desc)  # ← Archive handles competition
        
        logbook.update(generation, archive)
    
    return GenResult(target, archive.instances, logbook)
```

**Key Differences from EAGenerator:**
- ✅ Archive structure IS the population
- ✅ No explicit novelty algorithm
- ❌ Rigid archive type (GridArchive | CVTArchive only)
- ❌ Mutation-only variation (no crossover)

---

#### **DEAGenerator** - Dominated Evolutionary Algorithm

**Key Code:** [generators.py](generators.py#L523-L664)

```python
class DEAGenerator(EAGenerator):  # ← Inherits from EAGenerator
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        pop_size: int = 128,
        offspring_size: int = 128,  # ← Can differ
        generations: int = 1000,
        k: int = 15,  # ← DNS parameter
        descriptor_strategy: str = "features",
        # ... operators
    ):
        super().__init__(...)  # Inherit EAGenerator infrastructure
        self.k = k
```

**Execution Loop:**
```python
def __call__(self, verbose: bool = False) -> GenResult:
    # 1. INITIALIZATION (same as EAGenerator)
    
    # 2. MAIN LOOP
    for gen in range(generations):
        # 2a. VARIATION (same as EAGenerator)
        offspring = _generate_offspring(pop_size)
        off_perf, off_scores = _evaluate_population(offspring)
        off_desc, off_feat = _update_descriptors(offspring, off_scores)
        
        # 2b. COMBINE population + offspring
        combined = np.concatenate([
            (descriptors, off_desc),
            (perf_biases, off_perf),
            # ... etc
        ])
        
        # 2c. APPLY DNS - Dynamic Local Competition
        (
            sorted_desc,
            sorted_perf,
            competition_fitness,  # ← DNS scores!
            sorted_idx
        ) = dominated_novelty_search(
            descriptors=combined_desc,
            performances=combined_perf,
            k=self.k,
            force_feasible_only=True
        )
        
        # 2d. SELECT TOP pop_size from combined (no replacement operator)
        keep_idx = sorted_idx[:pop_size]
        population = [combined[i] for i in keep_idx]
        
        logbook.update(generation, population)
    
    return GenResult(target, population, logbook)
```

**Dominated Novelty Search (DNS) Algorithm:**
```python
def dominated_novelty_search(descriptors, performances, k=15, force_feasible_only=True):
    """
    For each solution i:
    1. Find all solutions j where perf[j] > perf[i] (they dominate i)
    2. Among dominators, find k nearest neighbors
    3. Compute competition_fitness = avg distance to k dominators
    4. If no dominators exist → competition_fitness = ∞
    
    Returns: Solutions sorted descending by competition_fitness
    
    Intuition: Diverse solutions get ∞ (no competition), while
    similar lower-perf solutions must compete on how far from better ones.
    """
```

**Key Differences from EAGenerator:**
- ✅ No external archive (population IS output)
- ✅ Dynamic local competition (depends on neighborhood)
- ✅ For k-NN uses actual competitiveness, not abstract novelty
- ❌ Entirely replaces parent `__call__()` (no code reuse)

---

### What are their dependencies on archives, domains, operators, and transformers?

#### **EAGenerator Dependencies**

| Dependency | Required? | Type | Used For |
|------------|-----------|------|----------|
| **Domain** | ✅ Yes | ABC | `generate_instances()`, `generate_problems_from_instances()`, `extract_features()` |
| **Portfolio** | ✅ Yes | Iterable[SupportsSolve] | Solve all problems, rate each instance |
| **NS (Novelty Search)** | ✅ Yes | Object with `.archive`, `__call__()` | Compute diversity scores via k-NN |
| **Archive** | ⚠️ Optional | Any Archive subclass | Store novel solutions (if `solution_set` provided) |
| **Operators** | ✅ Yes (injected) | Callables (Crossover, Mutation, Selection, Replacement) | Genetic variation |
| **Transformer** | ⚠️ Optional | SupportsTransform | Reduce descriptor dimensionality |
| **PerformanceFn** | ✅ Yes (default: max_gap_target) | Callable | Compute performance bias from solver scores |

**Tight Coupling:**
```python
# Cannot bypass NS algorithm
self._novelty_search = novelty_approach  # Required, no default
```

#### **MapElitesGenerator Dependencies**

| Dependency | Required? | Type | Coupling Level |
|------------|-----------|------|-----------------|
| **Domain** | ✅ Yes | ABC | Same as EAGenerator |
| **Portfolio** | ✅ Yes | Iterable[SupportsSolve] | Same as EAGenerator |
| **Archive** | ✅ Yes | GridArchive \| CVTArchive | 🔴 TIGHTLY TYPED |
| **Mutation** | ✅ Yes (only) | Callable | No crossover allowed |
| **Descriptor** | ✅ Yes | String key in DESCRIPTORS dict | 🔴 FRAGILE (string lookup) |

**🔴 Critical Coupling:**
```python
if not isinstance(archive, (GridArchive, CVTArchive)):
    raise ValueError(...)  # Runtime type check!
```

#### **DEAGenerator Dependencies**

| Dependency | Required? | Type | Coupling Level |
|------------|-----------|------|-----------------|
| **Domain** | ✅ Yes | ABC | Inherited from EAGenerator |
| **Portfolio** | ✅ Yes | Iterable[SupportsSolve] | Inherited from EAGenerator |
| **Operators** | ✅ Yes (crossover, mutation, selection) | Callables | Inherited from EAGenerator |
| **NS** | ❌ No! | Ignored | Uses `dominated_novelty_search()` function instead |
| **Archive** | ❌ No | Not used | Population replaces archive |

**Cleaner Design:**
```python
# Removed NS requirement completely
# Uses functional DNS instead of class-based NS
```

---

### What coupling exists between generators and other components?

#### Coupling Matrix (Current State)

```
                 EAGenerator    MapElites    DEAGenerator
Domain           🔴 Tight       🔴 Tight      🔴 Tight
Portfolio        ✅ Injected    ✅ Injected   ✅ Injected
NS/Archive       🔴 Hard        🔴 Hard       ✅ Implicit
Operators        ✅ Injected    ⚠️ Partial    ✅ Inherited
Transformers     ✅ Optional    N/A          ✅ Inherited
DescriptorStrat  ⚠️ Mixed       🔴 String    ✅ Inherited
PerformanceFn    ✅ Injected    ✅ Injected   ✅ Inherited
```

**Legend:** 
- 🔴 Tight: Hard-coded dependencies, cannot swap
- ⚠️ Mixed: Partially coupled
- ✅ Injected: Constructor argument, easily replaceable

#### Specific Coupling Issues

**Issue 1: EAGenerator tightly coupled to NS object**
```python
# Cannot change diversity algorithm without creating new generator
def __init__(self, ..., novelty_approach: NS):  # No alternative!
    self._novelty_search = novelty_approach
```

**Issue 2: Fitness hardcoded to phi-blended formula**
```python
def __compute_fitness(self, perf_bias, novelty_scores):
    fitness = (fitness * self.phi) + (novelty_scores * phi_r)
    # No way to use different fitness composition (e.g., multi-objective)
```

**Issue 3: Domain tightly coupled in all generators**
```python
# Cannot work without domain's specific methods
self.domain.generate_instances(n=self.pop_size)
self.domain.generate_problems_from_instances(population)
self.domain.extract_features(population)

# If domain interface changes, all generators break!
```

**Issue 4: Descriptor strategy via string lookup (MapElites)**
```python
descriptor: str = "features"
self._descriptor_strategy = DESCRIPTORS[descriptor]  # Hidden global dict!
```

**Issue 5: MapElites rejects any archive not exactly GridArchive or CVTArchive**
```python
if not isinstance(archive, (GridArchive, CVTArchive)):
    raise ValueError(...)
# Even a compatible archive would be rejected!
```

---

## Question 2: Archives

### What is the Archive base class interface?

**Location:** [archives/_base_archive.py](archives/_base_archive.py#L1-L160)

```python
class Archive:
    """Stores a collection of diverse Instances"""
    
    def __init__(
        self,
        threshold: float,
        instances: Optional[Sequence[Instance]] = None,
        dtype=np.float64,
    ):
        self._storage = {"instances": [], "descriptors": []}
        self._threshold = threshold  # Minimum sparseness to include
    
    @property
    def instances(self) -> Sequence[Instance]:
        return self._storage["instances"]  # All stored solutions
    
    @property
    def descriptors(self) -> np.ndarray:
        return np.asarray(self._storage["descriptors"])  # Feature matrix
    
    @property
    def threshold(self):
        return self._threshold
    
    def extend(
        self,
        instances: Sequence[Instance],
        novelty_scores: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None,
    ):
        """Add instances with sparseness > threshold"""
        # ⚠️ Implementation NOT shown in base class!
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def __iter__(self):
        return iter(self._storage["instances"])
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)(self._threshold, self.instances[key])
        return self._storage["instances"][key]
```

**Critical Issue:** `extend()` is NOT abstract! No enforced contract.

---

### How do GridArchive and CVTArchive extend it?

#### **GridArchive** - Regular n-dimensional grid

**Location:** [archives/_grid_archive.py](archives/_grid_archive.py#L1-L200)

```python
class GridArchive(Archive):
    """Divides measure space into regular cells"""
    
    def __init__(
        self,
        dimensions: Sequence[int],  # cells per dimension, e.g., [20, 30]
        ranges: Sequence[Tuple[float, float]],  # bounds, e.g., [(0,10), (0,20)]
        instances: Optional[Iterable[Instance]] = None,
        eps: float = 1e-6,  # precision for bin boundaries
    ):
        Archive.__init__(self, threshold=np.finfo(np.float32).max)
        
        self._dimensions = np.asarray(dimensions)  # [20, 30]
        self._lower_bounds = ...  # [0, 0]
        self._upper_bounds = ...  # [10, 20]
        self._interval = ...  # [10, 20]
        
        # Precompute boundaries: [20, 30] cells define 21, 31 boundaries
        _bounds = []
        for dim, lb, ub in zip(self._dimensions, self._lower_bounds, self._upper_bounds):
            _bounds.append(np.linspace(lb, ub, dim))  # Cell boundaries
        self._boundaries = np.asarray(_bounds)
        
        # One elite per cell
        self._grid: Dict[int, np.ndarray] = {}  # flat_cell_id -> elite_index
        self._storage: Dict[int, Instance] = {}  # flat_cell_id -> Instance
    
    @property
    def coverage(self) -> float:
        """Filled cells / total cells"""
        return len(self._grid) / np.prod(self._dimensions)
    
    @property
    def n_cells(self) -> int:
        return np.prod(self._dimensions)  # Total available cells
    
    def extend(self, instances, descriptors, novelty_scores=None):
        """Add best instance per cell"""
        # For each (instance, descriptor):
        #   1. Compute cell_index = digitize(descriptor, boundaries)
        #   2. If cell empty OR instance better than current elite → add
```

**Key Properties:**
- **One solution per cell** (elite per region)
- **Capacity** = `dimensions[0] * dimensions[1] * ... * dimensions[n]`
- **Index computation:** `np.digitize(descriptors, boundaries)`

**Example:**
```python
# 20x30 grid, measure space [0,10] × [0,20]
archive = GridArchive(
    dimensions=[20, 30],
    ranges=[(0, 10), (0, 20)]
)
# Can hold UP TO 600 solutions (one per cell)
# coverage = num_filled_cells / 600
```

---

#### **CVTArchive** - Centroidal Voronoi Tessellation

**Location:** [archives/_cvt_archive.py](archives/_cvt_archive.py#L1-L200)

```python
class CVTArchive(GridArchive, RNG):  # 🔴 INHERITANCE IS WRONG
    """Divides space into k homogeneous Voronoi regions (k-means clusters)"""
    
    def __init__(
        self,
        k: int,  # number of centroids (regions)
        ranges: Sequence[Tuple[float, float]],
        n_samples: int,  # bootstrap samples for k-means
        centroids: Optional[npt.NDArray | str] = None,  # precomputed
        samples: Optional[npt.NDArray | str] = None,  # precomputed
    ):
        # Inherits GridArchive BUT changes behavior!
        GridArchive.__init__(self, dimensions=(1,) * len(ranges), ...)
        
        self._k = k  # number of regions
        self._n_samples = n_samples
        
        # K-means clustering to find centroids
        self._kmeans = KMeans(n_clusters=k, n_init=1)
        if self._samples is None:
            self._samples = rng.uniform(...)  # Random samples
        self._kmeans.fit(self._samples)
        self._centroids = self._kmeans.cluster_centers_
        
        # KD-tree for nearest centroid lookup
        self._kdtree = KDTree(self._centroids, metric="euclidean")
    
    def extend(self, instances, descriptors):
        """Add best instance per Voronoi region"""
        # For each (instance, descriptor):
        #   1. Find nearest centroid: centroid_id = kdtree.query(descriptor)
        #   2. If region empty OR instance better → add to region
```

**Key Differences:**
- **Adaptive partitioning** (k-means instead of uniform grid)
- **k regions** instead of dimensional grid
- **KDTree lookup** instead of binary search on boundaries
- **Can handle high-dimensional** spaces better than GridArchive

**Problem with Inheritance:**
```python
# CVTArchive IS-NOT-A GridArchive semantically!
# Yet inherits from it and changes core behavior
class CVTArchive(GridArchive, RNG):  # 🔴 Wrong
    # Better would be:
    # class CVTArchive(Archive, RNG):
    #     def __init__(self, k, ranges, n_samples):
```

---

### What methods must subclasses implement?

**Current Implementation:**
- Archive is NOT abstract (no `@abstractmethod`)
- No formal contract exists
- Subclasses "implement" by overriding `extend()`

**Methods Implied by Protocol:**

| Method | Required? | Signature | Purpose |
|--------|-----------|-----------|---------|
| `extend()` | ✅ Must override | `(instances, descriptors, novelty_scores=None)` | Add instances to archive |
| `__getitem__()` | ⚠️ Optional | `(index) -> Instance` or `(slice) -> Archive` | Support indexing |
| `__iter__()` | ⚠️ Optional | `() -> Iterator[Instance]` | Support iteration |
| `__len__()` | ⚠️ Optional | `() -> int` | Support `len()` |

**More Methods (Domain-specific):**

```python
# GridArchive only:
@property
def coverage(self) -> float: ...  # Filled cells / total cells

@property
def filled_cells(self) -> Keys: ...  # Set of non-empty cell indices

def index_of(self, descriptor) -> int: ...  # Cell index for descriptor

# CVTArchive only:
@property
def centroids(self) -> np.ndarray: ...  # k centroids

def samples(self) -> np.ndarray: ...  # Bootstrap samples
```

---

### How are archives used by generators?

#### **EAGenerator Usage Pattern**

```python
# 1. Optional: Create archive for novelty search
archive = Archive(threshold=0.001)
ns = NS(archive=archive, k=15)

# 2. Pass to generator
gen = EAGenerator(..., solution_set=archive)

# 3. In execution loop:
novelty_scores = ns(descriptors)  # ← Archive queried via NS
archive.extend(feasible_offspring, descriptors, novelty_scores)

# 4. Retrieve results
return GenResult(
    instances=ns.archive.instances,  # ← Output from archive
    ...
)
```

**Archive Role:** External storage for discovered novel solutions

---

#### **MapElitesGenerator Usage Pattern**

```python
# 1. Create structured archive BEFORE generator
archive = GridArchive(dimensions=[20, 30], ranges=[(0,10), (0,20)])

# 2. Pass to generator
gen = MapElitesGenerator(..., archive=archive)

# 3. In execution loop:
parents = archive[selected_indices]  # ← Archive gives parents
offspring = mutation(parents)
archive.extend(offspring, descriptors)  # ← Archive stores solutions

# 4. Retrieve results
return GenResult(
    instances=archive.instances,  # ← Output IS the archive
    ...
)
```

**Archive Role:** Population replacement (parents + fitness storage)

---

#### **DEAGenerator Usage Pattern**

```python
# No archive used!
gen = DEAGenerator(...)

# Results are the population directly
return GenResult(
    instances=self.population,  # ← Not from archive
    ...
)
```

**Archive Role:** None (implicit in population)

---

## Question 3: Domains

### What is the Domain interface/protocol?

**Location:** [_core/_domain.py](_core/_domain.py#L1-L200)

```python
class Domain(ABC, RNG):  # Abstract Base Class (not Protocol!)
    """Defines problem domain via abstract methods"""
    
    def __init__(
        self,
        dimension: int,
        bounds: Sequence[tuple],  # [(lb_0, ub_0), ..., (lb_n, ub_n)]
        dtype=np.float64,
        name: str = "Domain",
        feat_names: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        self._dimension = dimension
        self._bounds = bounds
        self._lbs = np.array(lower_bounds)  # Vector of lower bounds
        self._ubs = np.array(upper_bounds)  # Vector of upper bounds
    
    @abstractmethod
    def generate_instances(self, n: int = 1) -> List[Instance]:
        """Generate n random instances representing problems"""
        raise NotImplementedError()
    
    @abstractmethod
    def generate_problems_from_instances(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[Problem]:
        """Convert instances to solvable Problem objects"""
        raise NotImplementedError()
    
    @abstractmethod
    def extract_features(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> np.ndarray:
        """Extract descriptor features (shape: n_instances × n_features)"""
        raise NotImplementedError()
    
    @abstractmethod
    def extract_features_as_dict(
        self, instances: Sequence[Instance] | np.ndarray
    ) -> List[Dict[str, np.float32]]:
        """Return features as list of dicts with feature names as keys"""
        raise NotImplementedError()
```

**Domain Interface Enforced:**
```
Domain
├── generate_instances(n) → Instance[]
│   └── Creates random problem instances
├── generate_problems_from_instances(Instance[]) → Problem[]
│   └── Makes instances solvable
├── extract_features(Instance[]) → ndarray(n × m)
│   └── Computes descriptors for diversity
└── extract_features_as_dict(Instance[]) → Dict[]
    └── Same features with names
```

---

### How do BPP, KP, and TSP domains differ?

#### **BPP Domain** (Bin Packing Problem)

**Location:** [domains/bpp.py](domains/bpp.py#L1-L150)

```python
class BPP(Problem):  # ← Problem class (solvable problem instance)
    def __init__(self, items: Iterable[int], capacity: int):
        self._items = tuple(items)      # Item weights
        self._capacity = capacity
        # Instance variables: bin assignment for each item
        bounds = [(0, len(items)-1) for _ in range(len(items))]  # bin indices
    
    def evaluate(self, individual: Sequence) -> tuple[float]:
        """Falkenauer fitness: packing efficiency"""
        used_bins = np.max(individual) + 1
        fill_i = np.zeros(used_bins)
        for item, bin_id in enumerate(individual):
            fill_i[bin_id] += self._items[item]
        
        fitness = sum((f_i / capacity) ** 2 for f_i in fill_i) / used_bins
        return (fitness,)

class BPPDomain(Domain):  # ← Domain class (generates instances)
    def __init__(self, dimension: int = 50, capacity: int = 100):
        # Creates random BPP instances (random items + capacity)
        bounds = [(0, dimension-1) for _ in range(dimension)]
    
    def generate_instances(self, n: int) -> List[Instance]:
        """Generate n random BPP instances"""
        instances = []
        for _ in range(n):
            items = self._rng.integers(1, 100, size=self._dimension)
            capacity = int(np.sum(items) * 0.8)  # 80% fill target
            instances.append(Instance(variables=[capacity, *items]))
        return instances
    
    def generate_problems_from_instances(self, instances):
        """Convert to BPP objects (solvable)"""
        return [BPP(inst[1:], inst[0]) for inst in instances]
    
    def extract_features(self, instances):
        """Domain-specific features (shape: n × 3)"""
        features = np.zeros((len(instances), 3))
        for i, inst in enumerate(instances):
            items = inst[1:]
            features[i, 0] = np.mean(items)      # Average item size
            features[i, 1] = np.std(items)       # Item variance
            features[i, 2] = len(items) / inst[0]  # Item/capacity ratio
        return features
```

**BPP Instance Format:** `[capacity, weight_0, weight_1, ..., weight_n]`  
**BPP Solution:** Assignment vector `[bin_0, bin_1, ..., bin_n]`

---

#### **KP Domain** (0/1 Knapsack Problem)

**Location:** [domains/kp.py](domains/kp.py#L1-L200)

```python
class Knapsack(Problem):  # ← Problem instance
    def __init__(
        self,
        profits: Sequence[int],
        weights: Sequence[int],
        capacity: int,
    ):
        self.profits = profits
        self.weights = weights
        self.capacity = capacity
        self.penalty_factor = 100.0  # Constraint penalty
    
    def evaluate(self, individual: Sequence) -> Tuple[float]:
        """Profit minus capacity violation penalty"""
        profit = np.dot(individual, self.profits)
        packed = np.dot(individual, self.weights)
        penalty = max(0, packed - capacity) * penalty_factor
        return (profit - penalty,)

class KnapsackDomain(Domain):  # ← Domain class
    def __init__(
        self,
        dimension: int = 50,  # Number of items
        min_p: int = 1,
        max_p: int = 1000,
        min_w: int = 1,
        max_w: int = 1000,
        capacity_approach: str = "evolved",  # How capacity is set
        capacity_ratio: float = 0.8,
    ):
        # Mixed variables: capacity + (weight, profit) pairs
        bounds = [(1, max_capacity)] + [
            (min_w, max_w) if i % 2 == 0 else (min_p, max_p)
            for i in range(2 * dimension)
        ]
    
    def generate_instances(self, n: int) -> List[Instance]:
        """Generate n random KP instances"""
        instances = []
        for _ in range(n):
            # Random profits and weights
            profits = self._rng.integers(min_p, max_p, size=dimension)
            weights = self._rng.integers(min_w, max_w, size=dimension)
            
            # Capacity depends on approach
            if self._capacity_approach == "evolved":
                cap = self._rng.integers(0, max_capacity)
            elif self._capacity_approach == "percentage":
                cap = int(np.sum(weights) * capacity_ratio)
            else:  # "fixed"
                cap = max_capacity
            
            instances.append(Instance(variables=[cap, *zip(weights, profits)]))
        return instances
    
    def extract_features(self, instances):
        """KP-specific features (shape: n × 8)"""
        features = np.zeros((len(instances), 8))
        for i, inst in enumerate(instances):
            cap, items = inst[0], inst[1:]
            weights = items[::2]
            profits = items[1::2]
            
            features[i, 0] = cap  # Capacity
            features[i, 1] = len(items)  # Number of items
            features[i, 2] = np.mean(weights)
            features[i, 3] = np.std(weights)
            features[i, 4] = np.mean(profits)
            features[i, 5] = np.std(profits)
            features[i, 6] = np.mean(profits / weights)  # Efficiency
            features[i, 7] = np.sum(weights) / cap  # Fill ratio
        return features
```

**KP Instance Format:** `[capacity, w_0, p_0, w_1, p_1, ..., w_n, p_n]`  
**KP Solution:** Binary vector `[include_0, include_1, ..., include_n]` ∈ {0,1}^n

---

#### **TSP Domain** (Travelling Salesman Problem)

**Not fully shown in readings, but likely follows pattern:**

```python
class TSPDomain(Domain):
    def __init__(self, dimension: int = 50):  # Number of cities
        self._dimension = dimension
        bounds = [(0, 1)] * (2 * dimension)  # x, y coordinates
    
    def generate_instances(self, n: int):
        """Random cities in [0,1]^2"""
        # Format: [x_0, y_0, x_1, y_1, ..., x_n, y_n]
    
    def generate_problems_from_instances(self, instances):
        """Convert to distance matrices"""
        # Compute Euclidean distances: TSP(distance_matrix)
    
    def extract_features(self, instances):
        """Route statistics: city clustering, scale, etc."""
        # Spatial features (shape: n × k)
```

**TSP Instance Format:** City coordinates `[x_0, y_0, x_1, y_1, ..., x_n, y_n]`

---

### What methods do they implement?

**Common to all three (inherit from Domain ABC):**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `generate_instances()` | `(n: int) → Instance[]` | Create n random problem instances |
| `generate_problems_from_instances()` | `(Instance[]) → Problem[]` | Make instances solvable |
| `extract_features()` | `(Instance[]) → ndarray` | Compute descriptors |
| `extract_features_as_dict()` | `(Instance[]) → Dict[]` | Named features |

**Domain-Specific Differences:**

```
BPP               KP                TSP
├─ items/bins     ├─ items/capacity  ├─ cities/routes
├─ packing eff.   ├─ profit/weight   ├─ total distance
├─ bin stats      ├─ efficiency      ├─ city clustering
└─ ...            └─ fill ratio      └─ ...
```

**No code is shared** between BPPDomain, KnapsackDomain, and TSPDomain!
- Each implements abstract methods from scratch
- No inheritance hierarchy (all directly inherit Domain)
- Problem: Difficult to add new domain (copy-paste large amounts)

---

## Question 4: Operators

### What are the operator protocols?

**Location:** [operators/__init__.py](operators/__init__.py)

```python
# TYPE ALIASES (not formal Protocol classes)

Crossover = Callable[
    [IndType, IndType],
    IndType,
]
# Signature: (individual, other) → offspring

Mutation = Callable[
    [IndType, Sequence[Tuple]],
    IndType,
]
# Signature: (individual, bounds) → mutated_individual

Selection = Callable[
    [Sequence[IndType]],
    IndType,
]
# Signature: (population) → selected_individual

Replacement = Callable[
    [Sequence[IndType], Sequence[IndType]],
    Sequence[IndType],
]
# Signature: (current_population, offspring) → new_population
```

**Note:** These are simple type aliases, NOT formal Protocol classes. Better would be:

```python
# Proposed (better):
from typing import Protocol

class Crossover(Protocol):
    def __call__(self, ind1: IndType, ind2: IndType) -> IndType: ...

class Mutation(Protocol):
    def __call__(self, individual: IndType, bounds: Sequence[Tuple]) -> IndType: ...

class Selection(Protocol):
    def __call__(self, population: Sequence[IndType]) -> IndType: ...

class Replacement(Protocol):
    def __call__(
        self,
        current: Sequence[IndType],
        offspring: Sequence[IndType]
    ) -> Sequence[IndType]: ...
```

---

### How are they injected into generators?

#### **Constructor Injection Pattern**

```python
class EAGenerator(Generator, RNG):
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        novelty_approach: NS,
        pop_size: int = 100,
        # ... operators INJECTED:
        crossover: Crossover = uniform_crossover,
        mutation: Mutation = uniform_one_mutation,
        selection: Selection = binary_tournament_selection,
        replacement: Replacement = generational_replacement,
        # ...
    ):
        # Store as instance attributes
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.replacement = replacement
```

#### **Usage in Execution Loop**

```python
def _generate_offspring(self, offspring_size: int) -> np.ndarray:
    offspring = [None] * offspring_size
    for i in range(offspring_size):
        # 1. Selection
        parent_1 = self.selection(self.population)  # ← Injected!
        parent_2 = self.selection(self.population)
        
        # 2. Crossover
        child = self.__reproduce(parent_1, parent_2)
        offspring[i] = child
    return np.array(offspring)

def __reproduce(self, parent_1: Instance, parent_2: Instance) -> Instance:
    offspring = parent_1.clone()
    if self._rng.random() < self.cxrate:
        # Apply crossover operator
        offspring = self.crossover(offspring, parent_2)  # ← Injected!
    
    # Apply mutation operator
    offspring = self.mutation(offspring, self.domain.bounds)  # ← Injected!
    return offspring

def __call__(self, verbose=False):
    # ...
    # Apply replacement operator
    self.population = self.replacement(self.population, offspring)  # ← Injected!
```

**Pattern: Clean Dependency Injection**
- ✅ All operators are parameters
- ✅ Default implementations provided
- ✅ Easy to swap implementations
- ✅ No hardcoded algorithm coupling

---

## Question 5: Overall Architecture

### How tightly coupled are components?

**Coupling Scorecard (out of 10, lower=better):**

| Component Pair | Coupling Score | Issue |
|----------------|----------------|-------|
| Generator ↔ Domain | 🔴 9/10 | Domain method signatures required |
| EAGenerator ↔ NS | 🔴 9/10 | Required parameter, no abstraction |
| MapElites ↔ Archive | 🔴 8/10 | Runtime type checking for GridArchive\|CVTArchive |
| Generator ↔ Operators | ✅ 2/10 | Fully injected, completely decoupled |
| Generator ↔ Portfolio | ✅ 2/10 | Only calls SupportsSolve protocol |
| Domain ↔ Problem | 🟠 6/10 | Specific problem classes per domain |
| Archive ↔ Instance | ✅ 3/10 | Only stores, doesn't care about structure |
| CVTArchive ↔ GridArchive | 🔴 9/10 | Wrong inheritance hierarchy |

**Overall Assessment: MODERATE coupling (5/10)**
- Generators are tightly bound to domains
- Operators are excellently decoupled
- Archives have type checking issues
- No central registry or factory pattern

---

### What patterns are used for abstraction?

#### Pattern 1: Protocol-Based Interface
```python
class Generator(Protocol):
    def __call__(self, ...) -> GenResult: ...
    def _update_descriptors(...) -> Tuple: ...

# Enables structural subtyping - any class with these methods works
```

#### Pattern 2: Type Aliases for Operators
```python
Crossover = Callable[[IndType, IndType], IndType]
Mutation = Callable[[IndType, Sequence[Tuple]], IndType]

# Simple but lacks IDE support compared to Protocol classes
```

#### Pattern 3: Mix-in for Shared Functionality
```python
class Domain(ABC, RNG):  # RNG mix-in adds random generation
class CVTArchive(GridArchive, RNG):  # RNG mix-in

# Clean separation of concerns
```

#### Pattern 4: Strategy Pattern with Defaults
```python
class EAGenerator:
    def __init__(
        self,
        crossover: Crossover = uniform_crossover,  # ← Default
        mutation: Mutation = uniform_one_mutation,  # ← Default
    ):
        self.crossover = crossover
        self.mutation = mutation
```

#### Pattern 5: Dataclass Results Container
```python
@dataclass
class GenResult:
    target: str
    instances: np.ndarray
    history: Logbook
    metrics: Optional[pd.Series] = None

# Immutable result wrapping
```

#### Pattern 6: String-Based Configuration (Anti-pattern!)
```python
descriptor_strategy: str = "features"
self._descriptor_strategy = DESCRIPTORS[self._desc_key]

# 🔴 Fragile, no IDE support, silent failures
```

---

### What makes it hard to add new generators/archives/domains?

#### ❌ Adding a new Generator

**Barrier: 60% Code Duplication**

All three generators (`EAGenerator`, `MapElitesGenerator`, `DEAGenerator`) share:
- `_evaluate_population()` - Identical evaluation loop
- `_update_descriptors()` - Nearly identical descriptor extraction
- Instance evaluation pattern

**Current approach:** Copy-paste code across generators

**Needed:** Extract to `BaseGenerator` class

```python
# Proposed:
class BaseGenerator(Generator, RNG):
    def _evaluate_population(self, population):
        # Shared code for evaluating all instances
        ...
    
    def _extract_descriptors(self, population, scores):
        # Shared descriptor extraction
        ...

class MyNewGenerator(BaseGenerator):
    def __call__(self, verbose=False):
        # Only new algorithm logic
        ...
```

---

#### ❌ Adding a new Archive

**Barriers:**
1. **No ArchiveProtocol** - Runtime type checking in MapElites
2. **CVTArchive wrong inheritance** - Inherits GridArchive despite different semantics
3. **Implicit contract** - No abstract methods to enforce interface

**Current approach:** Subclass Archive or GridArchive

**Solution:** Define formal ArchiveProtocol

```python
# Proposed:
class ArchiveProtocol(Protocol):
    @property
    def instances(self) -> Sequence[Instance]: ...
    
    @property
    def descriptors(self) -> np.ndarray: ...
    
    def extend(self, instances, descriptors, scores=None): ...
    
    def __len__(self) -> int: ...
    
    def __iter__(self): ...

# Then use in generators:
class MapElitesGenerator:
    def __init__(self, archive: ArchiveProtocol):  # Type-safe!
        ...
```

Then runtime checks become unnecessary:
```python
# Current (fragile):
if not isinstance(archive, (GridArchive, CVTArchive)):
    raise ValueError(...)

# Better (Protocol-based):
# No check needed - structural typing handles it
```

---

#### ❌ Adding a new Domain

**Barriers:**
1. **No shared base logic** - BPP, KP, TSP all implement from scratch
2. **4 abstract methods** required for each domain
3. **No infrastructure** for instance validation, feature caching
4. **Problem classes required** - Must create BPP, Knapsack, etc. alongside domain

**Current process (tedious):**
```python
# 1. Create Problem class
class MyProblem(Problem):
    def __init__(self, ...):
        ...
    def evaluate(self, individual):
        ...

# 2. Create Domain class
class MyDomain(Domain):
    def generate_instances(self, n):
        ...
    def generate_problems_from_instances(self, instances):
        ...
    def extract_features(self, instances):
        ...
    def extract_features_as_dict(self, instances):
        ...

# ~200 lines of duplicative code
```

**Solution:** Template/Factory pattern or utility library

```python
# Proposed:
class DomainFactory:
    @staticmethod
    def create_domain(
        problem_type: str,
        dimension: int,
        bounds: Sequence[Tuple],
        feature_extractors: Dict[str, Callable],
    ) -> Domain:
        """Factory reduces boilerplate"""
        ...
```

---

#### ⚠️ Adding a New Operator (Actually Easy!)

```python
# Simple! No barriers!
def my_blend_crossover(ind1, ind2, alpha=0.3):
    offspring = ind1.clone()
    for i in range(len(ind1)):
        offspring[i] = (1 - alpha) * ind1[i] + alpha * ind2[i]
    return offspring

# Use immediately:
gen = EAGenerator(..., crossover=my_blend_crossover)
```

**Why easy?** Operators are just callables, fully injected.

---

## Summary: Tight Coupling Clusters

### 🔴 Cluster 1: Generator → Domain
- **Issue:** Every generator needs domain's specific methods
- **Cannot swap:** Domain type checking happens at runtime
- **If domain interface changes:** All generators break

### 🔴 Cluster 2: Archive Type System
- **Issue:** MapElites hardcodes `(GridArchive | CVTArchive)`
- **No abstraction:** String-based descriptor lookup
- **Runtime checks:** Should use Protocols

### 🟡 Cluster 3: Code Duplication
- **Issue:** 60% shared code across generators (unused)
- **Maintenance burden:** Same evaluation logic in 3 places
- **No base extraction:** Forces copy-paste patterns

### ✅ Operator System (OK)
- **Decoupled:** Fully injected, no hardcoding
- **Extensible:** Just define callable with right signature
- **No barriers:** Can add new operators trivially

---

## Recommended Refactoring Priority

| Priority | Change | Effort | Impact |
|----------|--------|--------|--------|
| 🔴 High | Extract `BaseGenerator` for shared evaluation | 2 days | Eliminate 60% duplication |
| 🔴 High | Define `ArchiveProtocol` to replace type checks | 1 day | Enable custom archives |
| 🟠 Medium | Move descriptor lookup to `Enum` | 1 day | Better type safety |
| 🟠 Medium | CVTArchive → composition instead of inheritance | 1 day | Cleaner design |
| 🟡 Low | Add formal `OperatorProtocol` classes | 4 hours | IDE support, documentation |
| 🟡 Low | Domain factory/utilities | 3 days | Reduce domain boilerplate |

