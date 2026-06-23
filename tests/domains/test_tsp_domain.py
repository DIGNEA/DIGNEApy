#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_tsp_domain.py
@Time    :   2025/03/05 14:37:47
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import platform
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_equal

from digneapy.core import Instance
from digneapy.domains.tsp import TSP, TSPDomain


@pytest.mark.parametrize("save_as_1d", (True, False))
def test_tsp_problem_attrs(save_as_1d):
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(
        number_of_nodes=number_of_nodes,
        coords=coordinates,
        save_distances_as_1d=save_as_1d,
    )

    assert len(tsp) == number_of_nodes
    assert_equal(tsp.coordinates, coordinates)

    # The computation of the distances was not delayed
    assert tsp.distances is not None
    assert isinstance(tsp.distances, np.ndarray)
    # TSP stores the distance matrix as a flattened 1D
    # array with only the upper diagonal elements
    if save_as_1d:
        n_expected_distances = ((number_of_nodes - 1) * (number_of_nodes)) // 2
        assert len(tsp.distances) == n_expected_distances
    else:
        assert_equal(tsp.distances.shape, (number_of_nodes, number_of_nodes))


@pytest.mark.parametrize("save_as_1d", (True, False))
def test_tsp_problem_attrs_created_from_list(save_as_1d):
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = (
        np.random
        .default_rng()
        .uniform(low=low, high=high, size=(number_of_nodes, 2))
        .tolist()
    )
    assert isinstance(coordinates, list)
    assert not isinstance(coordinates, np.ndarray)
    tsp = TSP(
        number_of_nodes=number_of_nodes,
        coords=coordinates,
        save_distances_as_1d=save_as_1d,
    )

    assert len(tsp) == number_of_nodes
    assert_equal(tsp.coordinates, coordinates)

    # The computation of the distances was not delayed
    assert tsp.distances is not None
    assert isinstance(tsp.distances, np.ndarray)
    # TSP stores the distance matrix as a flattened 1D
    # array with only the upper diagonal elements
    if save_as_1d:
        n_expected_distances = ((number_of_nodes - 1) * (number_of_nodes)) // 2
        assert len(tsp.distances) == n_expected_distances
    else:
        assert_equal(tsp.distances.shape, (number_of_nodes, number_of_nodes))


def test_tsp_problem_attrs_with_postponed_dist_comp():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(
        number_of_nodes=number_of_nodes, coords=coordinates, postpone_dist_comp=True
    )

    assert len(tsp) == number_of_nodes
    assert_equal(tsp.coordinates, coordinates)

    # The computation of the distances was delayed
    assert_equal(tsp.distances, np.empty(0))


def test_tsp_problem_raises_with_wrong_args():
    with pytest.raises(ValueError):
        # Raises because we passed negative number_of_nodes
        number_of_nodes = 100
        low = 0.0
        high = 1000.0
        coordinates = np.random.default_rng().uniform(
            low=low, high=high, size=(number_of_nodes, 2)
        )
        _ = TSP(
            number_of_nodes=-number_of_nodes,
            coords=coordinates,
        )

    with pytest.raises(ValueError):
        # Raises because we passed negative penalty_factor
        number_of_nodes = 100
        low = 0.0
        high = 1000.0
        coordinates = np.random.default_rng().uniform(
            low=low, high=high, size=(number_of_nodes, 2)
        )
        _ = TSP(
            number_of_nodes=number_of_nodes, coords=coordinates, penalty_factor=-10.0
        )

    with pytest.raises(ValueError):
        # Raises because we passed invalid penalty_factor
        number_of_nodes = 100
        low = 0.0
        high = 1000.0
        coordinates = np.random.default_rng().uniform(
            low=low, high=high, size=(number_of_nodes, 2)
        )
        _ = TSP(
            number_of_nodes=number_of_nodes, coords=coordinates, penalty_factor="abc"
        )

    with pytest.raises(ValueError):
        # Raises because we passed coords as 1D
        number_of_nodes = 100
        low = 0.0
        high = 1000.0
        coordinates = (
            np.random
            .default_rng()
            .uniform(low=low, high=high, size=(number_of_nodes, 2))
            .flatten()
        )
        _ = TSP(
            number_of_nodes=number_of_nodes,
            coords=coordinates,
        )

    with pytest.raises(ValueError):
        # Raises because we passed double number of coords
        number_of_nodes = 100
        low = 0.0
        high = 1000.0
        coordinates = np.random.default_rng().uniform(
            low=low, high=high, size=(number_of_nodes * 2, 2)
        )
        _ = TSP(
            number_of_nodes=number_of_nodes,
            coords=coordinates,
        )


def test_tsp_problem_can_be_casted_to_array():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    tsp_as_array = np.asarray(tsp)
    assert_equal(tsp_as_array, coordinates)


def test_tsp_problem_can_create_solution():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    sample_solution = tsp.create_solution()
    assert len(sample_solution) == number_of_nodes
    assert_equal(sample_solution, np.arange(number_of_nodes))
    assert len(sample_solution.constraints) == 1
    assert len(sample_solution.objectives) == 1


def test_tsp_problem_can_be_saved_to_file():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    filename = "testing_tsp_problem.tsp"
    tsp.to_file(filename=filename)
    filename = Path(filename)
    assert filename.exists()
    filename.unlink()


def test_tsp_problem_can_be_saved_with_Path():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    filename = Path("testing_tsp_problem.tsp")
    tsp.to_file(filename=filename)
    assert filename.exists()
    filename.unlink()


def test_tsp_problem_can_be_loaded_from_file():
    number_of_nodes = 100
    filename = Path(__file__).parent / "data" / "testing_tsp.tsp"
    tsp = TSP.from_file(filename)
    assert isinstance(tsp, TSP)
    assert len(tsp) == number_of_nodes
    assert isinstance(tsp.distances, np.ndarray)
    assert_equal(tsp.coordinates.shape, (number_of_nodes, 2))


def test_tsp_problem_can_be_casted_to_instance():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)

    instance = tsp.to_instance()
    # TSP instance is a flattened version of the coordinates matrix
    assert len(instance) == number_of_nodes * 2
    assert_equal(instance.variables, coordinates.flatten())


def test_tsp_problem_evaluates_correct_solution():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    solution = tsp.create_solution()
    fitness, duplicated = tsp.evaluate(solution)
    assert_equal(solution.fitness, fitness)
    assert fitness >= 0.0
    # No constraints should be violated with the sample solution
    assert_equal((duplicated), 0)
    assert_equal(solution.constraints, (0,))


def test_tsp_problem_evaluates_correct_solution_with_1d_matrix():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(
        number_of_nodes=number_of_nodes, coords=coordinates, save_distances_as_1d=True
    )
    solution = tsp.create_solution()
    fitness, duplicated = tsp.evaluate(solution)
    assert_equal(solution.fitness, fitness)
    assert fitness >= 0.0
    # No constraints should be violated with the sample solution
    assert_equal((duplicated), 0)
    assert_equal(solution.constraints, (0,))


def test_tsp_problem_evaluates_correct_solution_saved_as_1d_or_2d():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp_1d = TSP(
        number_of_nodes=number_of_nodes, coords=coordinates, save_distances_as_1d=True
    )
    solution = tsp_1d.create_solution()
    fitness, duplicated = tsp_1d.evaluate(solution)

    tsp_2d = TSP(
        number_of_nodes=number_of_nodes, coords=coordinates, save_distances_as_1d=False
    )
    solution_2d = tsp_2d.create_solution()
    fitness_2d, duplicated_2d = tsp_1d.evaluate(solution)

    assert_equal(solution, solution_2d)
    assert_equal(fitness, fitness_2d)
    # No constraints should be violated with the sample solution
    assert_equal((duplicated), (0))
    assert_equal((duplicated_2d), (0))


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Test not supported on macOS runners"
)
def test_tsp_problem_evaluates_correct_solution_large_to_fit():
    number_of_nodes = 100_000
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    solution = tsp.create_solution()
    fitness, duplicated = tsp.evaluate(solution)
    assert_equal(solution.fitness, fitness)
    assert fitness >= 0.0
    # No constraints should be violated with the sample solution
    assert_equal(duplicated, 0)
    assert_equal(solution.constraints, (0,))


def test_tsp_problem_evaluates_raises_if_len_mismatch():
    number_of_nodes = 100
    low = 0
    high = 1000
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    solution = np.random.default_rng().integers(
        low=low, high=high, size=(number_of_nodes + 2)
    )
    with pytest.raises(ValueError):
        _ = tsp.evaluate(solution)


def test_tsp_problem_evaluates_duplicated_nodes_solution():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    solution = tsp.create_solution()
    # Node 1 is visited more than once
    solution[2] = 1
    fitness, duplicated = tsp.evaluate(solution)
    assert_equal(solution.fitness, fitness)
    assert fitness >= 0.0
    # First constraint should violated with this updated sample solution
    assert_equal(duplicated, 1)
    assert_equal(solution.constraints, (1,))


def test_tsp_problem_call_evaluates_correct_solution():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    solution = tsp.create_solution()
    fitness, duplicated = tsp(solution)
    assert_equal(solution.fitness, fitness)
    assert fitness >= 0.0
    # No constraints should be violated with the sample solution
    assert_equal(duplicated, 0)
    assert_equal(solution.constraints, (0,))


def test_tsp_problem_evaluates_with_sequence():
    number_of_nodes = 100
    low = 0.0
    high = 1000.0
    coordinates = np.random.default_rng().uniform(
        low=low, high=high, size=(number_of_nodes, 2)
    )
    tsp = TSP(number_of_nodes=number_of_nodes, coords=coordinates)
    solution = tsp.create_solution().variables
    fitness, duplicated = tsp.evaluate(solution)

    assert fitness >= 0.0
    assert_equal(duplicated, 0)


################ TSP Domain tests


def test_tsp_domain_attrs():
    number_of_nodes = 100
    domain = TSPDomain(number_of_nodes)
    expected_dimension = number_of_nodes * 2

    assert len(domain) == expected_dimension
    assert domain.bounds == [(0, 1000) for _ in range(expected_dimension)]


def test_tsp_domain_raises_negative_non():
    number_of_nodes = 100
    with pytest.raises(ValueError):
        _ = TSPDomain(number_of_nodes=-number_of_nodes)


def test_tsp_domain_raises_x_range_dim_not_2():
    number_of_nodes = 100
    with pytest.raises(ValueError):
        _ = TSPDomain(number_of_nodes=number_of_nodes, x_range=(0, 100, 1000))


def test_tsp_domain_raises_y_range_dim_not_2():
    number_of_nodes = 100
    with pytest.raises(ValueError):
        _ = TSPDomain(number_of_nodes=number_of_nodes, y_range=(0, 100, 1000))


def test_tsp_domain_raises_overlapped_x_range():
    number_of_nodes = 100
    with pytest.raises(ValueError):
        _ = TSPDomain(number_of_nodes=number_of_nodes, x_range=(100, 99))


def test_tsp_domain_raises_overlapped_y_range():
    number_of_nodes = 100
    with pytest.raises(ValueError):
        _ = TSPDomain(number_of_nodes=number_of_nodes, y_range=(100, 99))


def test_tsp_domain_can_extract_features():
    number_of_nodes = 100
    n_instances = 128
    n_features = 11
    domain = TSPDomain(number_of_nodes=number_of_nodes)

    instances = domain.generate_instances(n=n_instances)
    assert all(isinstance(x, Instance) for x in instances)
    features = domain.extract_features(instances)
    # The extract_features method cast the instances to np.ndarray
    # but it shouldn't alter the original
    assert all(isinstance(x, Instance) for x in instances)

    assert isinstance(features, np.ndarray)
    assert_equal(features.shape, (n_instances, n_features))
    # The last two could be zero depending of the instance
    assert (features[:, :-2] != 0.0).all()
    assert_equal(features[:, 0], number_of_nodes)


def test_tsp_domain_can_extract_features_as_dict():
    number_of_nodes = 100
    n_instances = 128
    n_features = 11
    domain = TSPDomain(number_of_nodes)

    instances = domain.generate_instances(n=n_instances)
    assert all(isinstance(x, Instance) for x in instances)
    features = domain.extract_features_as_dict(instances)
    expected_keys = "size,std_distances,centroid_x,centroid_y,radius,fraction_distances,area,variance_nnNds,variation_nnNds,cluster_ratio,mean_cluster_radius"
    expected_keys = set(expected_keys.split(","))
    # The extract_features method cast the instances to np.ndarray
    # but it shouldn't alter the original
    assert all(isinstance(x, Instance) for x in instances)
    assert isinstance(features, list)
    assert all(
        expected_keys == set(instance_features.keys()) for instance_features in features
    )
    assert all(len(f) == n_features for f in features)


def test_tsp_domain_can_generate_instances():
    number_of_nodes = 100
    n_instances = 128
    domain = TSPDomain(number_of_nodes)

    instances = domain.generate_instances(n=n_instances)
    assert all(isinstance(x, Instance) for x in instances)
    assert all(len(x) == number_of_nodes * 2 for x in instances)


def test_tsp_domain_can_generate_problems_from_instances():
    number_of_nodes = 100
    n_instances = 128
    domain = TSPDomain(number_of_nodes)

    instances = domain.generate_instances(n=n_instances)
    assert all(isinstance(x, Instance) for x in instances)
    assert all(len(x) == number_of_nodes * 2 for x in instances)

    problems = domain.generate_problems_from_instances(instances)
    assert all(isinstance(p, TSP) for p in problems)
    assert all(len(p) == number_of_nodes for p in problems)
    for i, problem in enumerate(problems):
        expected_coordinates = np.asarray(instances[i]).reshape(number_of_nodes, 2)
        assert_equal(problem.coordinates, expected_coordinates)
        assert_equal(problem.coordinates.shape, (number_of_nodes, 2))
