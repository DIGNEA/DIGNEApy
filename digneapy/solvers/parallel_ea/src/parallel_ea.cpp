/**
 * @file par_ea_kp.cpp
 * @author Alejandro Marrero (amarrerd@ull.edu.es)
 * @brief
 * @version 0.1
 * @date 2024-06-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <omp.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iterator>
#include <tuple>
#include <vector>

#include "PseudoRandom.h"
#include "pybind11/numpy.h"
namespace py = pybind11;
using namespace std;

/**
 * @brief Individual structure for a solution of the KP
 *
 */
struct Individual {
    float fitness;
    float constraint;
    vector<char> x;
};
float evaluateConstraints(Individual &solution, const vector<int> &weights,
                          const int capacity) {
    int packed = 0.0;
    // Asumimos que es factible y la factibilidad es cero
    int diff = 0;
    for (int i = 0; i < solution.x.size(); i++) {
        packed += weights[i] * solution.x[i];
    }
    diff = packed - capacity;
    int penalty = 100 * diff;
    if (penalty < 0) {
        penalty = 0;
    }
    solution.constraint = (float)penalty;
    return (float)penalty;
}

double evaluateKnapsack(Individual &solution, const vector<int> &weights,
                        const vector<int> &profits, const int capacity) {
    float fitness = 0;
    float penalty = evaluateConstraints(solution, weights, capacity);
    for (int i = 0; i < solution.x.size(); i++) {
        fitness += solution.x[i] * profits[i];
    }
    // Restamos la posible penalizacion
    fitness -= penalty;
    solution.fitness = fitness;
    return fitness;
}

Individual createKPSolution(const int n) {
    Individual solution{0.0, 0.0, {}};
    vector<char> vars(n, false);
    for (int i = 0; i < n; i++) {
        if (PseudoRandom::randDouble() > 0.5) {
            vars[i] = true;
        }
    }
    solution.x = vars;
    return solution;
}

class ParallelGeneticAlgorithm {
   public:
    ParallelGeneticAlgorithm(int populationSize, int generations,
                             double mutationRate, double crossRate,
                             int numberOfCores);

    virtual ~ParallelGeneticAlgorithm() = default;

    std::tuple<vector<int>, float> run(const int &n, const vector<int> &weights,
                                       const vector<int> &profits,
                                       const int capacity);

   protected:
    void createInitialPopulation(const int &n, const vector<int> &weights,
                                 const vector<int> &profits,
                                 const int capacity);

    void reproduction(Individual &, Individual &);

   protected:
    int populationSize;
    int generations;
    double mutationRate;
    double crossRate;
    int numberOfCores; /*!< Number of cores to run in parallel */
    int chunks;        /*!< Chunks of population for each core */
    vector<Individual> individuals;
};

ParallelGeneticAlgorithm::ParallelGeneticAlgorithm(int populationSize,
                                                   int generations,
                                                   double mutationRate,
                                                   double crossRate,
                                                   int numberOfCores)
    : populationSize(populationSize),
      generations(generations),
      mutationRate(mutationRate),
      crossRate(crossRate),
      numberOfCores(numberOfCores) {
    this->individuals.resize(this->populationSize, {0.0, 0.0, {}});
    this->chunks = this->populationSize / this->numberOfCores;
    omp_set_num_threads(this->numberOfCores);
}

void ParallelGeneticAlgorithm::createInitialPopulation(
    const int &n, const vector<int> &weights, const vector<int> &profits,
    const int capacity) {
    const int popDim = this->populationSize;
    this->individuals.resize(popDim);
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < popDim; i++) {
        this->individuals[i] = createKPSolution(n);
        evaluateKnapsack(individuals[i], weights, profits, capacity);
    }
}

std::tuple<vector<int>, float> ParallelGeneticAlgorithm::run(
    const int &n, const vector<int> &weights, const vector<int> &profits,
    const int capacity) {
    createInitialPopulation(n, weights, profits, capacity);
    const float eps = 1e-9;
    const int popDim = this->populationSize - 1;
    for (int g = 0; g < this->generations; g++) {
        vector<Individual> offspring(this->populationSize);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < this->populationSize; i++) {
            // Previously we splitted the population by cores
            // Splits thepopulation in chunks to perform the selection
            // int init = omp_get_thread_num() * chunks;
            // int end = init + (chunks - 1);
            int idx1 = min(PseudoRandom::randInt(0, popDim),
                           PseudoRandom::randInt(0, popDim));
            int idx2 = min(PseudoRandom::randInt(0, popDim),
                           PseudoRandom::randInt(0, popDim));
            Individual child1 = individuals[idx1];
            Individual child2 = individuals[idx2];
            this->reproduction(child1, child2);
            evaluateKnapsack(child1, weights, profits, capacity);
            // Replacement
            bool update = false;
            double childPenalty = child1.constraint;
            double individualPenalty = individuals[i].constraint;
            double childFitness = child1.fitness;
            double individualFitness = individuals[i].fitness;
            // If child has less penalty or in equal penalty values it has
            // better fitness
            if (childPenalty < individualPenalty) {
                update = true;
            } else if ((abs(childPenalty - individualPenalty) < eps) &&
                       (childFitness > individualFitness)) {
                update = true;
            }
            if (update) {
                offspring[i] = child1;
                // individuals[i] = child1;
            } else {
                offspring[i] = individuals[i];
            }
        }
// Replacement
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < this->populationSize; i++) {
            this->individuals[i] = offspring[i];
        }
    }
    vector<Individual> filteredPopulation;
    std::copy_if(this->individuals.begin(), this->individuals.end(),
                 std::back_inserter(filteredPopulation),
                 [](const Individual &individual) {
                     return individual.constraint == 0.0;
                 });

    auto comparison = [&](Individual &firstInd, Individual &secondInd) -> bool {
        return firstInd.fitness > secondInd.fitness;
    };
    std::sort(filteredPopulation.begin(), filteredPopulation.end(), comparison);
    if (!filteredPopulation.empty()) {
        std::vector<int> chromosome;
        chromosome.reserve(filteredPopulation[0].x.size());

        std::transform(
            filteredPopulation[0].x.begin(), filteredPopulation[0].x.end(),
            std::back_inserter(chromosome), [](char c) { return (int)c; });

        return std::make_tuple(chromosome, filteredPopulation[0].fitness);
    } else {
        return std::make_tuple(vector<int>(), -1.0);
    }
}

void ParallelGeneticAlgorithm::reproduction(Individual &child1,
                                            Individual &child2) {
    if (PseudoRandom::randDouble() < this->crossRate) {
        // Usamos C++17 para no declarar el tipo del vector que obtenemos
        vector<char> secondIndVars = child2.x;
        vector<char> firstIndVars = child1.x;

        for (int i = 0; i < firstIndVars.size(); i++) {
            if (PseudoRandom::randDouble() < 0.5) {
                auto tmpVariable = secondIndVars[i];
                auto copyVar = firstIndVars[i];
                secondIndVars[i] = copyVar;
                firstIndVars[i] = tmpVariable;
            }
        }
        child1.x = firstIndVars;
        child2.x = secondIndVars;
    }
    if (PseudoRandom::randDouble() < this->mutationRate) {
        int varIndex = PseudoRandom::randInt(0, int(child1.x.size() - 1));
        char varNewValue = PseudoRandom::randInt(0, 1);
        child1.x[varIndex] = varNewValue;
    }
}

PYBIND11_MODULE(parallel_ea, m) {
    m.doc() = "Parallel EA for Knapsack Problems";
    m.def("get_max_threads", &omp_get_max_threads,
          "Returns max number of threads");
    m.def("set_num_threads", &omp_set_num_threads, "Set number of threads");
    py::class_<ParallelGeneticAlgorithm>(m, "_ParEACpp")
        .def(py::init<int, int, double, double, int>(),
             py::arg("populationSize"), py::arg("generations"),
             py::arg("mutationRate"), py::arg("crossRate"),
             py::arg("numberOfCores"))
        .def("run", &ParallelGeneticAlgorithm::run, py::arg("n"),
             py::arg("weights"), py::arg("profits"), py::arg("capacity"));
}