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
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <experimental/random>
#include <iostream>
#include <iterator>
#include <vector>

#include "pybind11/numpy.h"
namespace py = pybind11;
using namespace std;

struct Individual {
    float fitness;
    float constraint;
    vector<int> x;
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
    solution.constraint = penalty;
    return float(penalty);
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

class ParallelGeneticAlgorithm {
   public:
    ParallelGeneticAlgorithm(int populationSize, int maxEvaluations,
                             double mutationRate, double crossRate,
                             int numberOfCores);

    virtual ~ParallelGeneticAlgorithm() = default;

    double run(const int &n, const vector<int> &weights,
               const vector<int> &profits, const int capacity);

   protected:
    void createInitialPopulation(const int &n, const vector<int> &weights,
                                 const vector<int> &profits,
                                 const int capacity);

    void reproduction(Individual &, Individual &);

   protected:
    int populationSize;
    int maxEvals;
    double mutationRate;
    double crossRate;
    int numberOfCores; /*!< Number of cores to run in parallel */
    int chunks;        /*!< Chunks of population for each core */
    vector<Individual> individuals;
};

ParallelGeneticAlgorithm::ParallelGeneticAlgorithm(int populationSize,
                                                   int maxEvaluations,
                                                   double mutationRate,
                                                   double crossRate,
                                                   int numberOfCores)
    : populationSize(populationSize),
      maxEvals(maxEvaluations),
      mutationRate(mutationRate),
      crossRate(crossRate),
      numberOfCores(numberOfCores) {
    this->individuals.resize(this->populationSize, {0.0, 0.0, {}});
    this->chunks = this->populationSize / this->numberOfCores;
    omp_set_num_threads(this->numberOfCores);
    std::srand(std::time(nullptr));
}

void ParallelGeneticAlgorithm::createInitialPopulation(
    const int &n, const vector<int> &weights, const vector<int> &profits,
    const int capacity) {
    const int popDim = this->populationSize;
    this->individuals.resize(popDim);
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < popDim; i++) {
        vector<int> x(n, 0);
        for (int j = 0; j < n; j++) {
            x.push_back(std::experimental::randint(0, 1));
        }
        this->individuals[i] = {0.0, 0.0, x};
        evaluateKnapsack(individuals[i], weights, profits, capacity);
    }
}

double ParallelGeneticAlgorithm::run(const int &n, const vector<int> &weights,
                                     const vector<int> &profits,
                                     const int capacity) {
    int performedEvaluations = 0;
    createInitialPopulation(n, weights, profits, capacity);
    const float eps = 1e-9;
    const int GENERATIONS = this->maxEvals / this->populationSize;
    int performedGenerations = 0;
    const int popDim = this->populationSize - 1;
    do {
        vector<Individual> offspring(this->populationSize);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < this->populationSize; i++) {
            // Previously we splitted the population by cores
            // Splits thepopulation in chunks to perform the selection
            // int init = omp_get_thread_num() * chunks;
            // int end = init + (chunks - 1);
            int idx1 = min(std::experimental::randint(0, popDim),
                           std::experimental::randint(0, popDim));
            int idx2 = min(std::experimental::randint(0, popDim),
                           std::experimental::randint(0, popDim));
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
        // Updating the performed evaluations
        performedGenerations++;
    } while (performedGenerations < GENERATIONS);
}

void ParallelGeneticAlgorithm::reproduction(Individual &child1,
                                            Individual &child2) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(gen) < this->crossRate) {
        // Usamos C++17 para no declarar el tipo del vector que obtenemos
        vector<int> secondIndVars = child2.x;
        vector<int> firstIndVars = child1.x;

        for (int i = 0; i < firstIndVars.size(); i++) {
            if (dis(gen) < 0.5) {
                auto tmpVariable = secondIndVars[i];
                auto copyVar = firstIndVars[i];
                secondIndVars[i] = copyVar;
                firstIndVars[i] = tmpVariable;
            }
        }
        child1.x = firstIndVars;
        child2.x = secondIndVars;
    }
    // this->mutation->run(child1, this->mutationRate, problem.get());
    if (dis(gen) < this->mutationRate) {
        int varIndex = std::experimental::randint(0, int(child1.x.size() - 1));
        int varNewValue = std::experimental::randint(0, 1);
        child1.x[varIndex] = varNewValue;
    }
}

PYBIND11_MODULE(par_ea_kp, m) {
    m.doc() = "Parallel EA for Knapsack Problems";
    m.def("get_max_threads", &omp_get_max_threads,
          "Returns max number of threads");
    m.def("set_num_threads", &omp_set_num_threads, "Set number of threads");
    py::class_<ParallelGeneticAlgorithm>(m, "ParEAKP")
        .def(py::init<int, int, double, double, int>())
        .def("run", &ParallelGeneticAlgorithm::run);
}