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

#include "pybind11/numpy.h"
namespace py = pybind11;
using namespace std;

#include <cstdlib>
#include <ctime>
#include <iostream>

/**
 *  This class is based on the random geneartor provided in the source
 *  code of Deb's implementation of NSGA-II.
 *  That implementation can be found in:
 *  http://www.iitk.ac.in/kangal/codes/nsga2/nsga2-v1.1.tar
 **/

class RandomGenerator {
   private:
    double seed_;
    double oldrand_[55];
    int jrand_;

    void randomize();

    void warmup_random(double seed);

    void advance_random();

    double randomperc();

   public:
    RandomGenerator(void);

    virtual ~RandomGenerator(void) = default;

    int rnd(int low, int high);

    double rndreal(double low, double high);

};  // RandomGenerator

/**
 *  This class is based on the random geneartor provided in the source
 *  code of Deb's implementation of NSGA-II.
 *  That implementation can be found in:
 *  http://www.iitk.ac.in/kangal/codes/nsga2/nsga2-v1.1.tar
 **/

RandomGenerator::RandomGenerator() : seed_(0.0), oldrand_(), jrand_() {
    srand(time(NULL));
    // srand(0);
    // cout << (unsigned)time(0) << endl;
    seed_ = ((double)rand() / (double)(RAND_MAX));
    // cout << "Seed value is: " << seed_ << endl;

    // seed_ = RAND_MAX;
    // cout << "Seed value is: " << seed_ << endl;

    // seed_ = (double) ((float) (float) seed_ / (float) RAND_MAX);
    // cout << "Seed value is: " << seed_ << endl;

    randomize();
}  // RandomGenerator

int RandomGenerator::rnd(int low, int high) {
    int res;
    if (low >= high) {
        res = low;
    } else {
        res = low + (int)(randomperc() * (high - low + 1));
        if (res > high) {
            res = high;
        }
    }
    return (res);
}

double RandomGenerator::rndreal(double low, double high) {
    return (low + (high - low) * randomperc());
}

void RandomGenerator::randomize() {
    int j1;

    for (j1 = 0; j1 <= 54; j1++) {
        oldrand_[j1] = .0;
    }  // for

    jrand_ = 0;
    warmup_random(seed_);
}

void RandomGenerator::warmup_random(double seed) {
    int j1, i1;
    double new_random, prev_random;
    oldrand_[54] = seed;
    new_random = 0.000000001;
    prev_random = seed;

    for (j1 = 1; j1 <= 54; j1++) {
        i1 = (21 * j1) % 54;
        oldrand_[i1] = new_random;
        new_random = prev_random - new_random;

        if (new_random < 0.0) {
            new_random += 1.0;
        }

        prev_random = oldrand_[i1];
    }

    advance_random();
    advance_random();
    advance_random();
    jrand_ = 0;

    return;
}

void RandomGenerator::advance_random() {
    int j1;
    double new_random;
    for (j1 = 0; j1 < 24; j1++) {
        new_random = oldrand_[j1] - oldrand_[j1 + 31];
        if (new_random < 0.0) {
            new_random = new_random + 1.0;
        }
        oldrand_[j1] = new_random;
    }
    for (j1 = 24; j1 < 55; j1++) {
        new_random = oldrand_[j1] - oldrand_[j1 - 24];
        if (new_random < 0.0) {
            new_random = new_random + 1.0;
        }
        oldrand_[j1] = new_random;
    }
}

double RandomGenerator::randomperc() {
    jrand_++;
    if (jrand_ >= 55) {
        jrand_ = 1;
        advance_random();
    }
    return oldrand_[jrand_];
}

#include <math.h>

#include <iostream>

/**
 * @brief This is the interface for the random number generator in dignea.
 * The idea is that all the random numbers will be generated using a single
 * random generator which will be accesible throug this interface.
 **/
class PseudoRandom {
   public:
    static RandomGenerator *randomGenerator_;

    PseudoRandom();

   public:
    /**
     * @brief Generates a random double value between 0.0 and 1.0
     *
     * @return double
     */
    static double randDouble();  //    static int randInt();
    /**
     * @brief Returns a random integer int the range [minBound, maxBound]
     *
     * @param minBound
     * @param maxBound
     * @return int
     */
    static int randInt(int minBound, int maxBound);
    /**
     * @brief Returns a random double in the range [minBound, maxBound]
     *
     * @param minBound
     * @param maxBound
     * @return double
     */
    static double randDouble(double minBound, double maxBound);
    /**
     * @brief Returns a random value extracted from a Normal Distribution with
     * mean and standardDeviation
     *
     * @param mean
     * @param standardDeviation
     * @return double
     */
    static double randNormal(double mean, double standardDeviation);
    /**
     * @brief Get random points from an hypersphere (center = 0, radius = 1)
     * Code taken from Maurice Clerc's implementations
     *
     * @param dimension
     * @return double*
     */
    static double *randSphere(int dimension);
};

/**
 * This file is aimed at defining the interface for the random generator.
 * The idea is that all the random numbers will be generated using a single
 * random generator which will be accesible throug this interface.
 **/

RandomGenerator *PseudoRandom::randomGenerator_ = nullptr;

PseudoRandom::PseudoRandom() {
    // randomGenerator_ = nullptr ;
    if (PseudoRandom::randomGenerator_ == nullptr) {
        PseudoRandom::randomGenerator_ = new RandomGenerator();
    }
}

// static int PseudoRandom::randInt() {
//     if (randomGenerator_ == nullptr) {
//         new PseudoRandom();
//     }
//     return randomGenerator_->rando
// }

double PseudoRandom::randDouble() {
    if (PseudoRandom::randomGenerator_ == nullptr) {
        PseudoRandom::randomGenerator_ = new RandomGenerator();
    }
    return PseudoRandom::randomGenerator_->rndreal(0.0, 1.0);
}

int PseudoRandom::randInt(int minBound, int maxBound) {
    if (PseudoRandom::randomGenerator_ == nullptr) {
        PseudoRandom::randomGenerator_ = new RandomGenerator();
    }
    return PseudoRandom::randomGenerator_->rnd(minBound, maxBound);
}

double PseudoRandom::randDouble(double minBound, double maxBound) {
    if (PseudoRandom::randomGenerator_ == nullptr) {
        PseudoRandom::randomGenerator_ = new RandomGenerator();
    }
    return PseudoRandom::randomGenerator_->rndreal(minBound, maxBound);
}

/**
 * Use the polar form of the Box-Muller transformation to obtain
 * a pseudo random number from a Gaussian distribution
 * Code taken from Maurice Clerc's implementation
 * @param mean
 * @param standardDeviation
 * @return A pseudo random number
 */
double PseudoRandom::randNormal(double mean, double standardDeviation) {
    double x1, x2, w, y1;

    do {
        x1 = 2.0 * randDouble() - 1.0;
        x2 = 2.0 * randDouble() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while (w >= 1.0);

    w = sqrt((-2.0 * log(w)) / w);
    y1 = x1 * w;
    y1 = y1 * standardDeviation + mean;
    return y1;
}

/**
 * Get a random point from an hypersphere (center = 0, radius = 1)
 * Code taken from Maurice Clerc's implementation
 * @param dimension
 * @return A pseudo random point
 */
double *PseudoRandom::randSphere(int dimension) {
    int D = dimension;
    double *x = new double[dimension];

    double length = 0;
    for (int i = 0; i < dimension; i++) x[i] = 0.0;

    // --------- Step 1. Direction

    for (int i = 0; i < D; i++) {
        x[i] = randNormal(0, 1);
        length += length + x[i] * x[i];
    }

    length = sqrt(length);

    // --------- Step 2. Random radius

    double r = randDouble(0, 1);

    for (int i = 0; i < D; i++) {
        x[i] = r * x[i] / length;
    }

    return x;
}

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