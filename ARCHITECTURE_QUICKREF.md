# DIGNEApy Architecture - Quick Reference Guide

## Component Quick Start

### 1. Generators at a Glance

| Generator | Algorithm | Diversity | Archive | Operators | Best For |
|-----------|-----------|-----------|---------|-----------|----------|
| **EAGenerator** | EA + Novelty Search | Explicit novelty scores | Custom Archive | Crossover, Mutation, Selection, Replacement | Exploring novel behaviors |
| **MapElitesGenerator** | Quality-Diversity (MAP-Elites) | Spatial grid/CVT divisions | GridArchive or CVTArchive | Mutation only | Coverage of behavior space |
| **DEAGenerator** | EA + Dominated Novelty Search | Dynamic local competition | Output list | Inherited from EAGenerator | Balancing performance & novelty |

### 2. Archive at a Glance

| Archive Type | Spatial Structure | Elite Selection | Best For |
|--------------|------------------|-----------------|----------|
| **Archive** (base) | Unstructured | Threshold-based | General collection |
| **GridArchive** | Regular n-dim grid | Best per cell | Regular measure spaces |
| **CVTArchive** | Voronoi cells (k-means) | Best per region | Adaptive partitioning |

### 3. Domain Implementations

| Domain | Instance Format | Problem Type | Feature Types |
|--------|-----------------|--------------|----------------|
| **BPP** | `[bin_0, bin_1, ..., bin_n]` | Bin Packing | Item-based features |
| **KP** | `[capacity, w_0, p_0, ..., w_n, p_n]` | 0/1 Knapsack | Efficiency metrics |
| **TSP** | Coordinates or distance matrix | Travelling Salesman | Distance-based |

---

## Code Patterns by Use Case

### Use Case 1: Run EAGenerator with Novelty Search

```python
from digneapy.generators import EAGenerator
from digneapy.domains import KnapsackDomain
from digneapy.archives import Archive
from digneapy._core import NS
from digneapy.operators import uniform_crossover, uniform_one_mutation

# 1. Create domain
domain = KnapsackDomain(dimension=50, capacity_approach="evolved")

# 2. Create portfolio (list of solvers)
portfolio = [solver_1, solver_2]  # Must implement SupportsSolve protocol

# 3. Create novelty search archive
ns_archive = Archive(threshold=0.0001)
ns_algorithm = NS(archive=ns_archive, k=15)

# 4. Create generator
gen = EAGenerator(
    domain=domain,
    portfolio=portfolio,
    novelty_approach=ns_algorithm,      # ← How diversity is computed
    pop_size=100,
    generations=500,
    descriptor_strategy="features",      # ← Uses domain.extract_features()
    phi=0.85,                            # ← Fitness blend: 85% perf, 15% novelty
    crossover=uniform_crossover,
    mutation=uniform_one_mutation,
    seed=42
)

# 5. Run generator
result = gen(verbose=True)

# 6. Access results
print(f"Generated {len(result.instances)} instances")
print(f"Best metrics:\n{result.metrics}")
print(f"Generations log:\n{result.history}")
```

**Key Dependencies:**
- `Domain` → generates instances & problem objects
- `Portfolio` → evaluates instances via solvers
- `NS` → computes novelty scores via k-NN on descriptors
- `Archive` → stores novel solutions
- `Operators` → genetic variation

---

### Use Case 2: Run MAP-Elites Generator

```python
from digneapy.generators import MapElitesGenerator
from digneapy.domains import BPPDomain
from digneapy.archives import GridArchive
from digneapy.operators import uniform_one_mutation
from digneapy._core._metrics import Statistics

# 1. Create domain
domain = BPPDomain(dimension=50, capacity=100)

# 2. Create structured archive (GridArchive)
# Imagine we want 2D feature space: [feature_0, feature_1]
# Feature space ranges: [0, 10] x [0, 20]
archive = GridArchive(
    dimensions=[20, 30],  # 20x30 = 600 cells
    ranges=[(0, 10), (0, 20)]  # bounds per dimension
)

# 3. Create generator (MAP-Elites uses mutation only, no crossover)
gen = MapElitesGenerator(
    domain=domain,
    portfolio=portfolio,
    initial_pop_size=100,
    generations=500,
    archive=archive,  # ← TYPE VALIDATED at runtime!
    mutation=uniform_one_mutation,
    repetitions=1,
    descriptor="features",  # ← String lookup in DESCRIPTORS
    seed=42
)

# 4. Run and analyze
result = gen(verbose=True)

# 5. Coverage metrics
print(f"Archive coverage: {archive.coverage * 100:.2f}%")
print(f"Filled cells: {len(archive.filled_cells)}")
print(f"Total solutions: {len(archive.instances)}")
```

**Key Insight:** MAP-Elites doesn't use explicit novelty search - archive structure handles diversity!

---

### Use Case 3: Run Dominated Novelty Search Generator

```python
from digneapy.generators import DEAGenerator

# DEAGenerator is like EAGenerator but uses Dominated Novelty Search
# instead of traditional novelty search
gen = DEAGenerator(
    domain=domain,
    portfolio=portfolio,
    pop_size=128,
    offspring_size=128,  # ← Can differ from population size
    generations=1000,
    k=15,  # ← k-nearest neighbors for DNS
    descriptor_strategy="features",
    seed=42
)

result = gen(verbose=True)
```

**Key Difference:** DNS creates dynamic local competition - individuals compete with better-performing neighbors, not global archive!

---

### Use Case 4: Add a Custom Operator

```python
# Crossover operators have signature: (Individual, Individual) -> Individual
def my_blend_crossover(ind1, ind2, alpha=0.3):
    """Blend random weighted mixture of genes"""
    offspring = ind1.clone()
    for i in range(len(ind1)):
        offspring[i] = (1 - alpha) * ind1[i] + alpha * ind2[i]
    return offspring

# Use it immediately!
gen = EAGenerator(
    domain=domain,
    portfolio=portfolio,
    novelty_approach=ns,
    crossover=my_blend_crossover,  # ← Injected!
    mutation=uniform_one_mutation,
    ...
)
```

**Decoupling Benefit:** No changes needed to generator code!

---

### Use Case 5: Custom Descriptor Strategy

**Hard way (current):**
```python
# Must modify DESCRIPTORS dict or rely on string lookup
descriptor_strategy="custom_features"
# Then somewhere in codebase:
DESCRIPTORS["custom_features"] = my_feature_extractor
```

**Problem:** String keys, silent failures, hard to trace.

**Cleaner Way (proposal):**
```python
# Define custom descriptor extractor
def extract_problem_difficulty(instances):
    """Extract difficulty features from instances"""
    features = np.zeros((len(instances), 3))
    for i, inst in enumerate(instances):
        # Domain-specific analysis
        features[i, 0] = compute_density(inst)
        features[i, 1] = compute_variance(inst)
        features[i, 2] = compute_correlation(inst)
    return features

# Pass directly (no string lookup needed)
from digneapy.transformers import SupportsTransform
gen = EAGenerator(
    domain=domain,
    portfolio=portfolio,
    novelty_approach=ns,
    descriptor_strategy="custom",
    custom_descriptor_fn=extract_problem_difficulty,  # ← Direct reference
    ...
)
```

---

## Architecture Decision Points

### Decision 1: Which Archive Type?

**Choose GridArchive if:**
- Feature space is well-understood
- Dimensions are few (2-4)
- Regular partitioning makes sense
- You know bounds in advance

**Choose CVTArchive if:**
- Feature space is high-dimensional
- Want adaptive partitioning
- Need k-means clustering
- Uncertain about region importance

**Choose Archive (base) if:**
- Using EAGenerator (no structured storage)
- Don't need spatial partitioning
- Just collecting diverse solutions

### Decision 2: Which Generator Type?

**Choose EAGenerator if:**
- Want explicit novelty search
- Need archival of novel solutions
- Can define good descriptor spaces
- Want performance + novelty balance

**Choose MapElitesGenerator if:**
- Coverage of behavior space is goal
- Have good 2-4D descriptors
- Performance varies by region
- Can afford GridArchive overhead

**Choose DEAGenerator if:**
- Want both performance & novelty
- Local competition preferred over global
- No explicit archive bounds needed
- Performance bias is priority

---

## Common Pitfalls & Solutions

### Pitfall 1: "My instances aren't improving!"

**Likely Causes:**
1. Portfolio has bad solvers → Check solver quality first
2. Descriptor strategy irrelevant → Use domain features
3. phi value too high (87% performance = low novelty pressure) → Increase novelty weight (reduce phi)
4. Archive threshold too high → Lower threshold
5. Operators wrong for problem → Verify bounds and Individual type

**Debugging:**
```python
# Check evaluation loop
gen = EAGenerator(...)
perf_bias, scores = gen._evaluate_population(gen.population)
print(f"Performance range: {perf_bias.min():.3f} to {perf_bias.max():.3f}")
print(f"Solver scores shape: {scores.shape}")

# Check descriptors
descriptors, features = gen._update_descriptors(gen.population)
print(f"Descriptor range: {descriptors.min():.3f} to {descriptors.max():.3f}")

# Check novelty scores
ns_scores = gen._novelty_search(descriptors)
print(f"Novelty scores: {ns_scores}")
```

### Pitfall 2: "Archive is way too sparse!"

**Likely Causes:**
1. Archive threshold too high (default 0.0001 is tight!) → Increase threshold
2. k value too large → Reduce k (fewer neighbors required)
3. Descriptor space too high-dimensional → Use transformer/PCA
4. Feature extraction outputs NaN → Check domain implementation

**Solution:**
```python
# Loosen archive threshold
archive = Archive(threshold=0.01)  # More permissive

# OR use adaptive threshold
archive = Archive(threshold=0.0)  # Accept everything initially

# OR reduce k
ns_algorithm = NS(archive=archive, k=5)  # Fewer neighbors
```

### Pitfall 3: "MapElites says my archive type is invalid!"

**Root Cause:** Runtime type checking!
```python
if not isinstance(archive, (GridArchive, CVTArchive)):
    raise ValueError(...)  # ← Rejects even compatible archives
```

**Solution (Current):**
Only use GridArchive or CVTArchive directly.

**Solution (Future):**
Use ArchiveProtocol with duck typing.

### Pitfall 4: "Descriptor lookup fails silently"

**Current behavior:**
```python
descriptor_strategy = "typo_features"
# Silent fallback to "instance" strategy
# No error raised, code continues mysteriously
```

**Workaround:**
```python
from digneapy._core.descriptors import DESCRIPTORS

# Validate before creating generator
valid_strategies = list(DESCRIPTORS.keys())
assert "features" in valid_strategies

gen = EAGenerator(..., descriptor_strategy="features")
```

---

## Performance Tips

### Tip 1: Batch Operations
```python
# Slow (individual mutations):
for i in range(pop_size):
    offspring[i] = uniform_one_mutation(offspring[i], bounds)

# Fast (vectorized mutation):
from digneapy.operators import batch_uniform_one_mutation
offspring = batch_uniform_one_mutation(offspring, lb, ub)
```

### Tip 2: Portfolio Redundancy
```python
# If solvers are similar, reduce repetitions
gen = EAGenerator(
    ...,
    repetitions=1,  # ← Single run per solver
    ...
)

# If solvers are diverse, increase repetitions
gen = EAGenerator(
    ...,
    repetitions=3,  # ← Average over 3 runs per solver
    ...
)
```

### Tip 3: Population Size Trade-off
```python
# Large population = better exploration, slower per generation
gen = EAGenerator(..., pop_size=1000, generations=100)

# Small population = fewer evaluations, need more generations
gen = EAGenerator(..., pop_size=100, generations=1000)

# Typical sweet spot: pop_size in [50, 200], generations in [500, 2000]
```

### Tip 4: Descriptor Dimensionality
```python
# If descriptors are >10D, use transformer
from digneapy.transformers import PCATransformer

transformer = PCATransformer(n_components=2)
transformer.train(initial_features)

gen = EAGenerator(
    ...,
    descriptor_strategy="features",
    transformer=transformer,  # ← Reduces to 2D
)
```

---

## File Organization for Understanding

**Start reading in this order:**

1. **[generators.py](generators.py#L60-L80)** - Protocol definition & GenResult
2. **[_core/_domain.py](_core/_domain.py#L1-L100)** - Domain ABC
3. **[archives/_base_archive.py](archives/_base_archive.py#L1-L50)** - Archive interface
4. **[operators/__init__.py](operators/__init__.py)** - Operator protocols
5. **[generators.py](generators.py#L75-L200)** - EAGenerator init & structure
6. **[generators.py](generators.py#L390-L500)** - MapElitesGenerator
7. **[generators.py](generators.py#L523-L664)** - DEAGenerator
8. **[domains/kp.py](domains/kp.py)** - Domain example (KnapsackDomain)
9. **[archives/_grid_archive.py](archives/_grid_archive.py)** - Archive implementation

---

## Extension Points (High → Low Priority)

| Priority | Component | Difficulty | Impact |
|----------|-----------|-----------|--------|
| **🔴 High** | Extract BaseGenerator class | Medium | Reduce 60% code duplication |
| **🔴 High** | Define ArchiveProtocol | Low | Enable runtime archive swapping |
| **🟠 Medium** | Enum-based descriptors | Low | Better type safety |
| **🟠 Medium** | Configuration objects | Medium | Avoid Dict lookups |
| **🟡 Low** | CVTArchive → composition | Medium | Better architecture |
| **🟡 Low** | More transformer types (PCA, VAE) | Medium | Better descriptor handling |
| **⚪ Optional** | Caching evaluation results | High | Performance optimization |

