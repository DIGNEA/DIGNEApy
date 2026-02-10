# Quick Navigation Guide: DIGNEApy Refactoring Documentation

Choose the document that matches your needs:

---

## 🎯 I want to understand the bigger picture

**Read**: [REFACTORING_STRATEGY.md](REFACTORING_STRATEGY.md)
- 10-phase refactoring plan
- What's wrong with current design
- New architecture overview
- Benefits breakdown
- Migration timeline
- Estimated effort

**Time**: 15 minutes

---

## 💻 I want to see code and start implementing

**Read**: [IMPLEMENTATION_TEMPLATES.md](IMPLEMENTATION_TEMPLATES.md)
- Ready-to-copy BaseGenerator class
- Protocol definitions for Archives/Domains
- Registry implementations
- Configuration system
- All templates ready to use in your codebase

**Time**: 20 minutes to read, 1-2 hours to implement one template

---

## 🔧 I want to add custom generators/archives/domains

**Read**: [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md)
- Full example: PSO-QD Generator (runnable code)
- Full example: k-NN Archive (runnable code)
- Full example: Custom TSP Domain (runnable code)
- Testing templates
- Distribution via pip

**Time**: 30 minutes to read, 1-2 hours to create your component

---

## 📊 I want to see concrete improvements

**Read**: [BEFORE_AND_AFTER.md](BEFORE_AND_AFTER.md)
- Current problems with examples
- New solutions with examples
- Side-by-side comparisons
- Concrete benefits
- Quick comparison table
- Backward compatibility info

**Time**: 15 minutes

---

## 📋 I need a checklist/summary

**Read**: This file + [REFACTORING_STRATEGY.md](REFACTORING_STRATEGY.md) Section 6 (Migration Guide)

**Checklist**:
```
Phase 1: Extract Common Logic
  ☐ Create _base_generator.py
  ☐ Move _evaluate_population() 
  ☐ Move _update_descriptors()
  ☐ Update existing generators to inherit

Phase 2: Define Protocols
  ☐ Create _archive_protocol.py
  ☐ Create _domain_protocol.py
  ☐ Update existing classes to satisfy protocols

Phase 3: Implement Registries
  ☐ Create _registry.py in generators/
  ☐ Create _registry.py in archives/
  ☐ Create _registry.py in domains/
  ☐ Register existing implementations

Phase 4: Configuration System
  ☐ Create config.py
  ☐ Implement GeneratorConfig
  ☐ Add tests

Phase 5: Examples & Documentation
  ☐ Update example scripts
  ☐ Add extension examples
  ☐ Publish plugin template
```

**Time to complete**: ~1 week

---

## 🚀 I want to start TODAY

**Option 1: Quick Win (30 min)**
1. Read [BEFORE_AND_AFTER.md](BEFORE_AND_AFTER.md) - Understand the improvements
2. Read [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md) Section "Using Custom Components"
3. You're ready to appreciate the benefits

**Option 2: Start Coding (1-2 hours)**
1. Read [IMPLEMENTATION_TEMPLATES.md](IMPLEMENTATION_TEMPLATES.md) - Template 1 (BaseGenerator)
2. Copy Template 1 code into `digneapy/generators/_base_generator.py`
3. Update `EAGenerator` to inherit from it (minimal changes)
4. You've done Phase 1!

**Option 3: Add Custom Component (2-4 hours)**
1. Read [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md) - Pick PSO, k-NN, or TSP example
2. Copy the example code into a new file in your codebase
3. Register it with @Registry.register("name") decorator
4. You now have a custom component!

---

## 📚 Full Reading Order (Recommended)

1. **BEFORE_AND_AFTER.md** (15 min)
   - See the problems and benefits in action

2. **REFACTORING_STRATEGY.md** Sections 1-3 (15 min)
   - Understand the overall architecture

3. **IMPLEMENTATION_TEMPLATES.md** (20 min)
   - See the code structure

4. **EXTENSION_GUIDE.md** (30 min)
   - See real examples

5. **REFACTORING_STRATEGY.md** Sections 6-10 (15 min)
   - Migration path and timeline

**Total**: ~90 minutes to full understanding

Then implement at your own pace!

---

## 🎓 Learn by Example

### Example 1: Quick Extension (30 min)
Goal: Add a custom generator

```bash
1. Read: EXTENSION_GUIDE.md Part 1 (PSO Generator)
2. Copy code to: digneapy/generators/pso_generator.py
3. Update: digneapy/generators/__init__.py
4. Test: python -c "from digneapy import GeneratorRegistry; print(GeneratorRegistry.list_generators())"
5. Done! ✅
```

### Example 2: Full Refactor (1 week)
Goal: Complete refactoring

```bash
1. Read: REFACTORING_STRATEGY.md
2. Implement Phase 1: BaseGenerator
3. Implement Phase 2: Protocols
4. Implement Phase 3: Registries
5. Implement Phase 4: Config
6. Implement Phase 5: Examples
7. Run all tests
8. Done! ✅
```

### Example 3: Enable Community (2 days)
Goal: Set up for plugins

```bash
1-6. Complete Full Refactor (Example 2)
7. Read: EXTENSION_GUIDE.md Part 5 (Distribution)
8. Create setup.py for your first plugin
9. Test pip install
10. Publish 🎉
```

---

## 🎯 By Role

### I'm a Library Maintainer
**Read order:**
1. REFACTORING_STRATEGY.md (full understanding)
2. IMPLEMENTATION_TEMPLATES.md (implementation guide)
3. EXTENSION_GUIDE.md (community enablement)

### I'm a User of DIGNEApy
**Read order:**
1. BEFORE_AND_AFTER.md (understand benefits)
2. EXTENSION_GUIDE.md (how to use/extend)
3. IMPLEMENTATION_TEMPLATES.md (reference if needed)

### I'm Contributing a Custom Generator
**Read order:**
1. EXTENSION_GUIDE.md (full example)
2. IMPLEMENTATION_TEMPLATES.md (registry details)

### I'm Researching QD Architecture
**Read order:**
1. REFACTORING_STRATEGY.md (full architecture)
2. BEFORE_AND_AFTER.md (comparison with Pyribs patterns)
3. IMPLEMENTATION_TEMPLATES.md (implementation details)

---

## ❓ FAQ

**Q: Does this break existing code?**
A: No! See BEFORE_AND_AFTER.md final section. Both old and new APIs work.

**Q: How long does refactoring take?**
A: See REFACTORING_STRATEGY.md Section 10. ~1 week for full implementation.

**Q: Can I do it incrementally?**
A: Yes! Each phase is independent. Do Phase 1, then 2, etc.

**Q: Will my custom generators work?**
A: Yes! Either with old API or new API. See examples in EXTENSION_GUIDE.md.

**Q: How do I add my generator as a plugin?**
A: See EXTENSION_GUIDE.md Part 5. Uses pip entry_points.

**Q: Is this similar to Pyribs?**
A: Similar plugin flexibility, different design. See REFACTORING_STRATEGY.md Section 1.

---

## 🔗 Document Structure

```
BEFORE_AND_AFTER.md
├─ Problems with current design
├─ Solutions with new design
└─ Side-by-side comparisons
    (Start here for motivation)

REFACTORING_STRATEGY.md
├─ Current issues (detail)
├─ Proposed architecture (complete plan)
├─ Why each change
├─ Benefits breakdown
├─ Migration path (step-by-step)
└─ Timeline & effort
    (Read for full understanding)

IMPLEMENTATION_TEMPLATES.md
├─ BaseGenerator (ready-to-use code)
├─ Protocols (ready-to-use code)
├─ Registries (ready-to-use code)
├─ Configuration (ready-to-use code)
└─ Integration guide
    (Copy-paste these templates)

EXTENSION_GUIDE.md
├─ Custom Generator Example (PSO)
├─ Custom Archive Example (k-NN)
├─ Custom Domain Example (TSP)
├─ Test examples
└─ Distribution guide
    (Follow these to extend)
```

---

## 💡 Pro Tips

- **Start small**: Do Phase 1 (BaseGenerator) first. Biggest impact, minimal risk.
- **Have tests**: Add tests for each phase. See EXTENSION_GUIDE.md Part 5.
- **Backward compatible**: Existing API unchanged. New API optional.
- **Community enabled**: Easy for others to contribute once done.
- **Reusable patterns**: Registries can be used in other projects too!

---

## 📞 Questions?

Each document has sections answering common questions:
- REFACTORING_STRATEGY.md → Section 8 (Benefits)
- IMPLEMENTATION_TEMPLATES.md → Comments in code
- EXTENSION_GUIDE.md → Testing section
- BEFORE_AND_AFTER.md → Quick Comparison Table

**Good luck with the refactoring! 🚀**
