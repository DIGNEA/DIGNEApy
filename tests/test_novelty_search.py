#!/usr/bin/env python

"""Tests for `digneapy` package."""

import pytest
import copy
import numpy as np
from digneapy.novelty_search import (
    Archive,
    NoveltySearch,
    _instance_descriptor_strategy,
)
from digneapy.core import Instance


def transformer(l):
    """
    Dummy transformer that takes a List and returns a list of list with two random numbers each
    """
    return [[np.random.rand(), np.random.rand()] for _ in range(len(l))]


@pytest.fixture
def default_archive():
    instances = [Instance(variables=list(range(d, d + 5))) for d in range(10)]
    return Archive(instances)


@pytest.fixture
def empty_archive():
    return Archive()


def test_empty_archive(empty_archive):
    assert 0 == len(empty_archive)


def test_archive_to_array(default_archive):
    np_archive = np.array(default_archive)
    assert len(np_archive) == len(default_archive)
    assert isinstance(np_archive, np.ndarray)


def test_iterable_default_archive(default_archive):
    descriptors = [Instance(range(d, d + 5)) for d in range(10)]
    assert len(descriptors) == len(default_archive)
    assert all(a == b for a, b in zip(descriptors, default_archive))
    assert all(a == b for a, b in zip(iter(descriptors), iter(default_archive)))


def test_not_equal_archives(default_archive, empty_archive):
    assert default_archive != empty_archive


def test_equal_archives(default_archive):
    a1 = copy.copy(default_archive)
    assert default_archive == a1


def test_append_instance(empty_archive):
    assert 0 == len(empty_archive)
    instance = Instance(variables=list(range(100)))
    empty_archive.append(instance)
    assert 1 == len(empty_archive)
    assert [instance] == empty_archive.instances
    d = list(range(10))
    with pytest.raises(Exception):
        empty_archive.append(d)


def test_extend_iterable(empty_archive, default_archive):
    assert 0 == len(empty_archive)
    d = default_archive.instances
    empty_archive.extend(d)
    assert len(empty_archive) == len(default_archive)
    assert empty_archive == default_archive


def test_bool_on_empty_archive(empty_archive):
    assert not empty_archive


def test_bool_on_default_archive(default_archive):
    assert default_archive


def test_archive_magic(default_archive):
    assert (
        default_archive.__str__()
        == f"Archive with 10 instances -> {str(tuple(default_archive))}"
    )
    duplicated = copy.deepcopy(default_archive)
    assert hash(duplicated) == hash(default_archive)


def test_archive_access(default_archive):
    assert len(default_archive) == 10
    assert type(default_archive[0]) == Instance
    assert len(default_archive[:2]) == 2
    with pytest.raises(IndexError):
        default_archive[100]


def test_archive_format(default_archive):
    assert (
        "(Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()), Instance(f=0.0,p=0.0, s=0.0, descriptor=()))"
        == format(
            default_archive,
        )
    )


def test_archive_repr(default_archive):
    assert "Archive(['', '', '', '', '', '', '', '', '', ''])" == format(
        repr(default_archive),
    )


@pytest.fixture
def nsf():
    return NoveltySearch(k=3, descriptor="features")


@pytest.fixture
def nsp():
    return NoveltySearch(k=3, descriptor="performance")


@pytest.fixture
def nsi():
    return NoveltySearch(k=3, descriptor="instance")


def test_default_nsf(nsf):
    assert nsf.t_a == 0.001
    assert nsf.t_ss == 0.001
    assert nsf.k == 3
    assert nsf._describe_by == "features"
    assert len(nsf.archive) == 0
    assert len(nsf.solution_set) == 0
    assert (
        nsf.__str__()
        == "NS(desciptor=features,t_a=0.001,t_ss=0.001,k=3,len(a)=0,len(ss)=0)"
    )
    assert (
        nsf.__repr__()
        == "NS<desciptor=features,t_a=0.001,t_ss=0.001,k=3,len(a)=0,len(ss)=0>"
    )


def test_default_nsp(nsp):
    assert nsp.t_a == 0.001
    assert nsp.t_ss == 0.001
    assert nsp.k == 3
    assert nsp._describe_by == "performance"
    assert len(nsp.archive) == 0
    assert len(nsp.solution_set) == 0
    assert (
        nsp.__str__()
        == "NS(desciptor=performance,t_a=0.001,t_ss=0.001,k=3,len(a)=0,len(ss)=0)"
    )
    assert (
        nsp.__repr__()
        == "NS<desciptor=performance,t_a=0.001,t_ss=0.001,k=3,len(a)=0,len(ss)=0>"
    )


def test_default_nsi(nsi):
    assert nsi.t_a == 0.001
    assert nsi.t_ss == 0.001
    assert nsi.k == 3
    assert nsi._describe_by == "instance"
    assert len(nsi.archive) == 0
    assert len(nsi.solution_set) == 0
    assert (
        nsi.__str__()
        == "NS(desciptor=instance,t_a=0.001,t_ss=0.001,k=3,len(a)=0,len(ss)=0)"
    )
    assert (
        nsi.__repr__()
        == "NS<desciptor=instance,t_a=0.001,t_ss=0.001,k=3,len(a)=0,len(ss)=0>"
    )


def test_default_nsf_by_features():
    ns = NoveltySearch(descriptor="a_brand_new_descriptor")
    assert ns._describe_by == "features"


def __random_descriptors(n, size: int = 100):
    return [np.random.uniform(low=0, high=100, size=n) for _ in range(size)]


@pytest.fixture
def random_population():
    features = __random_descriptors(n=10)
    performances = __random_descriptors(n=(4, 4))
    instances = [
        Instance(variables=np.random.randint(low=0, high=100, size=100))
        for _ in range(100)
    ]
    for i, instance in enumerate(instances):
        instance.features = features[i]
        instance.portfolio_scores = performances[i]

    return instances


def test_run_nsf(nsf, random_population):
    assert nsf._describe_by == "features"
    assert all(len(instance.features) != 0 for instance in random_population)
    sparseness = nsf.sparseness(random_population)
    assert len(sparseness) == len(random_population)

    # Here we check that the NS includes the novel_ta amount of
    # instances that are supposed to has a s >= t_a
    novel_ta = sum(1 for i in sparseness if i >= nsf.t_a)
    nsf._update_archive(random_population)
    assert len(nsf.archive) == novel_ta

    nsf._update_solution_set(random_population)
    assert len(nsf.solution_set) != 0

    current_len = len(nsf.archive)
    nsf._update_archive(list())
    assert current_len == len(nsf.archive)

    current_len = len(nsf.solution_set)
    nsf._update_solution_set(list())
    assert current_len == len(nsf.solution_set)

    # If empty population it should raise
    with pytest.raises(Exception):
        nsf.sparseness([])
    # If len(pop) < k it should raise
    with pytest.raises(Exception):
        nsf.sparseness(random_population[:3])

    # Here we check the sparseness calculation on the solution set
    spars_ss = nsf.sparseness_solution_set(random_population)
    assert len(spars_ss) == len(spars_ss)
    # Raises because the list is empty
    with pytest.raises(AttributeError):
        nsf.sparseness_solution_set(list())
    # Raises because the one element of the list is empty
    with pytest.raises(AttributeError):
        new_pop = random_population + [[]]
        nsf.sparseness_solution_set(new_pop)
    # Raises because we need at least to elements to calculate the sparseness
    with pytest.raises(AttributeError):
        nsf.sparseness_solution_set(random_population[:1])


def test_run_nsf_with_transformer(random_population):
    nsft = NoveltySearch(k=3, descriptor="features", transformer=transformer)
    assert all(len(instance.features) != 0 for instance in random_population)
    sparseness = nsft.sparseness(random_population)
    assert len(sparseness) == len(random_population)


def test_run_nsp(nsp, random_population):
    assert nsp._describe_by == "performance"
    assert all(len(instance.portfolio_scores) != 0 for instance in random_population)
    sparseness = nsp.sparseness(random_population)
    assert len(sparseness) == len(random_population)

    # Here we check that the NS includes the novel_ta amount of
    # instances that are supposed to has a s >= t_a
    novel_ta = sum(1 for i in sparseness if i >= nsp.t_a)
    nsp._update_archive(random_population)
    assert len(nsp.archive) == novel_ta
    nsp._update_solution_set(random_population)
    assert len(nsp.solution_set) != 0

    current_len = len(nsp.archive)
    nsp._update_archive(list())
    assert current_len == len(nsp.archive)
    current_len = len(nsp.solution_set)
    nsp._update_solution_set(list())
    assert current_len == len(nsp.solution_set)

    # If empty population it should raise
    with pytest.raises(Exception):
        nsp.sparseness([])
    # If len(pop) < k it should raise
    with pytest.raises(Exception):
        nsp.sparseness(random_population[:3])


def test_run_nsf_with_transformer(random_population):
    nspt = NoveltySearch(k=3, descriptor="features", transformer=transformer)
    assert all(len(instance.features) != 0 for instance in random_population)
    sparseness = nspt.sparseness(random_population)
    assert len(sparseness) == len(random_population)


def test_run_ns_instance(nsi, random_population):
    assert nsi._describe_by == "instance"
    assert all(len(instance.features) != 0 for instance in random_population)
    assert all(len(instance.portfolio_scores) != 0 for instance in random_population)
    assert all(len(instance) != 0 for instance in random_population)

    # Sparseness is calculated with the instance
    sparseness = nsi.sparseness(random_population)
    assert len(sparseness) == len(random_population)
    # Here we check that the NS includes the novel_ta amount of
    # instances that are supposed to has a s >= t_a
    novel_ta = sum(1 for i in sparseness if i >= nsi.t_a)
    nsi._update_archive(random_population)
    assert len(nsi.archive) == novel_ta
    nsi._update_solution_set(random_population)
    assert len(nsi.solution_set) != 0

    current_len = len(nsi.archive)
    nsi._update_archive(list())
    assert current_len == len(nsi.archive)

    current_len = len(nsi.solution_set)
    nsi._update_solution_set(list())
    assert current_len == len(nsi.solution_set)

    # If empty population it should raise
    with pytest.raises(Exception):
        nsi.sparseness([])
    # If len(pop) < k it should raise
    with pytest.raises(Exception):
        nsi.sparseness(random_population[:3])


def test_run_nsf_with_transformer(random_population):
    nsit = NoveltySearch(k=3, descriptor="features", transformer=transformer)
    assert all(len(instance.features) != 0 for instance in random_population)
    sparseness = nsit.sparseness(random_population)
    assert len(sparseness) == len(random_population)
