# Before & After: DIGNEApy Refactoring Examples

This document shows concrete examples of how the plugin-based architecture improves extensibility.

---

## Example 1: Adding a New Generator

### ❌ BEFORE: Lots of Boilerplate & Duplication

Current approach requires reimplementing shared logic:

```python
# Current: Must copy _evaluate_population() logic from EAGenerator
class MyNewGenerator:
    def __call__(self, verbose: bool = False) -> GenResult:
        population = [self.domain.generate_instance() for _ in range(self.pop_size)]
        
        for gen_idx in range(self.generations):
            # ⚠️ DUPLICATE CODE - copied from EAGenerator._evaluate_population()
            performances = np.zeros(len(population), dtype=np.float64)
            for instance_idx, instance in enumerate(population):
                instance_performance = []
                for solver in self.portfolio:
                    for _ in range(self.repetitions):
                        result = solver.solve(instance)
                        instance_performance.append(result.best_obj)
                performances[instance_idx] = self.performance_function(
                    np.array(instance_performance)
                )
            # ⚠️ END DUPLICATE CODE
            
            # ⚠️ DUPLICATE CODE - copied from EAGenerator._update_descriptors()
            descriptors = np.array([
                DESCRIPTORS[self.descriptor_strategy](instance)
                for instance in population
            ])
            # ⚠️ END DUPLICATE CODE
            
            # Compute novelty...
            novelty_scores = self.novelty_search(descriptors)
            
            # Your custom algorithm logic (maybe 20% of code)
            # ...
```

**Problems:**
- 60% code copied from existing generators
- Hard to maintain (bug fixes must be applied in multiple places)
- Easy to introduce inconsistencies
- Error-prone

---

### ✅ AFTER: Clean Inheritance & Reuse

With `BaseGenerator`, you only write your algorithm:

```python
# NEW: Just inherit and implement your algorithm
@GeneratorRegistry.register("my-algorithm")
class MyNewGenerator(BaseGenerator):
    def __call__(self, verbose: bool = False) -> GenResult:
        population = [self.domain.generate_instance() for _ in range(self.pop_size)]
        
        for gen_idx in range(self.generations):
            # ✅ Reuse shared evaluation logic
            performances, descriptors = self._evaluate_population(population)
            
            novelty_scores = self.novelty_search(descriptors)
            
            # Your custom algorithm logic (100% unique code)
            fitness_scores = self._compute_fitness(performances, novelty_scores)
            
            # ... rest of your algorithm ...
            
            population = self._generate_offspring(population, fitness_scores)
        
        return GenResult(
            target=self.__class__.__name__,
            instances=np.array(self.archive.instances),
            history=self.logbook,
        )
```

**Benefits:**
- ✅ 40% less code
- ✅ Both implementations use same evaluation logic
- ✅ Bug fixes in _evaluate_population() help entire ecosystem
- ✅ Easier to understand (focus on algorithm, not boilerplate)

---

## Example 2: Using Different Archive Types

### ❌ BEFORE: Type Checks & Limited Composition

```python
# CURRENT: Generators have hardcoded archive type checks
class MapElitesGenerator:
    def __init__(self, archive, ...):
        # ⚠️ Runtime type checking - what if I create a custom compatible archive?
        if not isinstance(archive, (GridArchive, CVTArchive)):
            raise ValueError(f"Unsupported archive type: {type(archive)}")
        
        self.archive = archive

# Result: You can't use a custom archive, even if it has all required methods!
my_archive = CustomKNNArchive()  # Perfect Archive interface
gen = MapElitesGenerator(archive=my_archive)  # ❌ CRASH: "Unsupported archive type"
```

**Problems:**
- ❌ Can't use custom archives (even if they work perfectly)
- ❌ Tight coupling between generators and specific archive types
- ❌ No way to swap archives at runtime
- ❌ Adding new archives requires modifying generator code

---

### ✅ AFTER: Protocol-Based Composition

```python
# NEW: Generators accept anything implementing ArchiveProtocol
class MapElitesGenerator(BaseGenerator):
    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable,
        archive: ArchiveProtocol,  # ✅ Type annotation, not runtime check
        ...
    ):
        self.archive = archive
        # ✅ No isinstance check needed!

# Result: Works with any archive type!
my_archive = CustomKNNArchive()
gen = MapElitesGenerator(archive=my_archive)  # ✅ Works!

standard_archive = GridArchive(...)
gen = MapElitesGenerator(archive=standard_archive)  # ✅ Works!
```

**Benefits:**
- ✅ Works with any archive satisfying the protocol
- ✅ Custom archives are first-class citizens
- ✅ Runtime flexibility (swap archives in tests)
- ✅ IDE knows what methods are available (protocol definition)

---

## Example 3: Swapping Domains

### ❌ BEFORE: Tight Domain Coupling

```python
# CURRENT: Each generator is tied to domain approach
generator = EAGenerator(
    domain=KnapsackDomain(),  # Specific domain
    archive=archive,
)

results = generator()

# Want to try same algorithm on TSP? Must create new generator instance
generator2 = EAGenerator(
    domain=TSPDomain(),  # Different domain instance
    archive=archive,
)
```

**Problems:**
- Configuration is mixed into code
- Repeated boilerplate for each domain
- Hard to run experiments across domains

---

### ✅ AFTER: Configuration-Based Composition

```python
# NEW: Configuration encapsulates choices
config = GeneratorConfig(
    generator_type="ea",
    domain_type="kp",
    archive_type="grid",
    generations=100,
)

# Run on KP
gen = config.create()
results_kp = gen()

# Run on TSP with same configuration
config.domain_type = "tsp"
gen = config.create()
results_tsp = gen()

# Or use registry directly
for domain_name in ["kp", "tsp", "bpp"]:
    config.domain_type = domain_name
    gen = config.create()
    results = gen()
    print(f"{domain_name}: {len(results.instances)} instances generated")
```

**Benefits:**
- ✅ Configuration separate from code
- ✅ Easy to try combinations
- ✅ Declarative (what not how)
- ✅ Works with YAML/JSON for experiment configs

---

## Example 4: Discovering Available Components

### ❌ BEFORE: Must Read Documentation

```python
# Where do I find all generators?
# Need to check: generators.py, then look for all classes...
# Need to check documentation...
# Maybe there's a custom one in examples/?

# Basically: trial and error or deep code diving
```

---

### ✅ AFTER: Programmatic Discovery

```python
# Want to see all available generators?
from digneapy import GeneratorRegistry

print("Available generators:")
for name in GeneratorRegistry.list_generators():
    print(f"  - {name}")
# Output:
#   - ea
#   - map-elites
#   - dea
#   - pso-qd (custom!)

# Get generator class for inspection
gen_class = GeneratorRegistry.get("pso-qd")
print(gen_class.__doc__)

# Same for archives and domains
from digneapy import ArchiveRegistry, DomainRegistry

print("Available archives:", ArchiveRegistry.list_archives())
# Output: ['grid', 'cvt', 'knn']

print("Available domains:", DomainRegistry.list_domains())
# Output: ['bpp', 'kp', 'tsp', 'custom-tsp']
```

**Benefits:**
- ✅ Programmatic discovery
- ✅ Works with auto-completion
- ✅ Supports IDE tooltips
- ✅ No manual documentation needed

---

## Example 5: Sharing Custom Components

### ❌ BEFORE: Fork & Modify

```python
# To share your PSO-QD generator, users must:
# 1. Copy the file into their project
# 2. Import it manually
# 3. Hope it doesn't have version conflicts
# 4. If updating: manually re-copy updated file
```

---

### ✅ AFTER: Pip Package Distribution

```python
# Your package: setup.py
setup(
    name="digneapy-pso-qd",
    install_requires=["digneapy>=1.0.0"],
    entry_points={
        "digneapy.generators": [
            "pso-qd = digneapy_pso_qd:PSOQDGenerator",
        ],
    },
)

# Users just: pip install digneapy-pso-qd
# Then:
from digneapy import GeneratorRegistry

gen = GeneratorRegistry.create("pso-qd", ...)  # ✅ Automatically available!
```

**Benefits:**
- ✅ Proper distribution via pip
- ✅ Version management
- ✅ Auto-discovery
- ✅ Community ecosystem

---

## Example 6: Testing

### ❌ BEFORE: Coupled Tests

```python
# Test is tightly coupled to EAGenerator implementation
def test_ea_generator():
    generator = EAGenerator(
        domain=KnapsackDomain(),
        archive=GridArchive(...),
        portfolio=[solver1],
        generations=10,
    )
    
    results = generator()
    assert len(results.instances) > 0
    # BUT: What if I change internal implementation?
    # Test might break even though behavior is correct
```

---

### ✅ AFTER: Protocol-Based Tests

```python
# Test generator protocol compliance
def test_generator_protocol():
    gen = GeneratorRegistry.create("my-generator", ...)
    
    # These should work for ANY generator implementing BaseGenerator
    assert hasattr(gen, 'generations')
    assert hasattr(gen, '_evaluate_population')
    assert callable(gen)
    
    results = gen()
    
    # Check GenResult protocol
    assert hasattr(results, 'instances')
    assert hasattr(results, 'history')
    assert len(results.instances) > 0

# Test archive protocol compliance
def test_archive_protocol():
    archive = ArchiveRegistry.create("my-archive", ...)
    
    # These should work for ANY archive implementing ArchiveProtocol
    assert hasattr(archive, 'instances')
    assert hasattr(archive, 'descriptors')
    assert hasattr(archive, 'extend')
    assert hasattr(archive, 'index_of')

# Benefits:
# ✅ Same tests work for all implementations
# ✅ New implementations automatically get tested
# ✅ Tests focus on behavior, not implementation
```

---

## Example 7: Mixing Strategies

### ❌ BEFORE: Must Re-implement to Use Different Components

```python
# Want to use EA + CVT archive (instead of EA + Grid)?
# You had to look at EAGenerator code and figure it out

# Want to use novel archive with existing generator?
# Hard to know which generators support it

# Want to try all generator + archive combinations?
# Need to write a lot of boilerplate
```

---

### ✅ AFTER: Flexible Composition

```python
# Try many combinations easily
generators = ["ea", "map-elites", "dea", "pso-qd"]
archives = ["grid", "cvt", "knn"]
domains = ["bpp", "kp", "tsp"]

results_by_combo = {}

for gen_type in generators:
    for arch_type in archives:
        for dom_type in domains:
            config = GeneratorConfig(
                generator_type=gen_type,
                domain_type=dom_type,
                archive_type=arch_type,
                generations=50,
            )
            
            try:
                gen = config.create()
                results = gen()
                results_by_combo[(gen_type, arch_type, dom_type)] = results
                print(f"✅ {gen_type} + {arch_type} + {dom_type}: {len(results.instances)} instances")
            except Exception as e:
                print(f"❌ {gen_type} + {arch_type} + {dom_type}: {e}")

# Result: Automatic exploration of design space!
```

**Benefits:**
- ✅ Easy experimentation
- ✅ Systematic exploration
- ✅ Find best combinations
- ✅ Reproducible

---

## Quick Comparison Table

| Aspect | Before | After |
|--------|--------|-------|
| **Adding new generator** | Copy 60% of code | Inherit BaseGenerator, 100% unique code |
| **Using custom archive** | ❌ Type check fails | ✅ Protocol-based, just works |
| **Discovering components** | Read docs, grep code | `GeneratorRegistry.list_generators()` |
| **Testing** | Coupled to impl | Protocol-based, impl-agnostic |
| **Sharing components** | Copy files | `pip install` |
| **Experimenting** | Lots of boilerplate | Config-based composition |
| **Type safety** | Runtime errors | IDE support via protocols |
| **Code reuse** | Duplicated | Shared in BaseGenerator |
| **Extensibility** | Hard | Easy |
| **Maintenance** | Many places to fix bugs | Single source of truth |

---

## Migration Path: Keep Current API Working

All improvements are backward compatible!

```python
# OLD CODE - Still works exactly the same
from digneapy import EAGenerator, GridArchive, KnapsackDomain

domain = KnapsackDomain(n_items=50)
archive = GridArchive(dimensions=(10, 10), ranges=[(-1, 1), (-1, 1)])
solver = domain.get_solver("greedy")

gen = EAGenerator(
    domain=domain,
    archive=archive,
    portfolio=[solver],
    generations=100,
)

results = gen()

# NEW CODE - Use new capabilities when you want
from digneapy import GeneratorConfig

config = GeneratorConfig(
    generator_type="pso-qd",  # Your new custom generator!
    domain_type="kp",
    archive_type="knn",       # Your new custom archive!
    generations=100,
)

gen = config.create()
results = gen()
```

Both approaches work! Adopt at your own pace.
