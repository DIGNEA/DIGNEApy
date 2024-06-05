/**
 * @file pseudo_random.cpp
 * @author Alejandro Marrero (amarrerd@ull.edu.es)
 * @brief
 * @version 0.1
 * @date 2024-06-05
 *
 * @copyright Copyright (c) 2024
 *
 */

//
// Created by amarrero on 15/12/20.
//

#include "PseudoRandom.h"

using namespace std;

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