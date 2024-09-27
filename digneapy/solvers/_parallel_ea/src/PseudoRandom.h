/**
 * @file PseudoRandom.h
 * @author Alejandro Marrero (amarrerd@ull.edu.es)
 * @brief
 * @version 0.1
 * @date 2024-06-05
 *
 * @copyright Copyright (c) 2024
 *
 */

//  PseudoRandom.h
//
//  Author:
//       Esteban López-Camacho <esteban@lcc.uma.es>
//       Antonio J. Nebro <antonio@lcc.uma.es>
//       Juan J. Durillo <durillo@lcc.uma.es>
//
//  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <math.h>

#include <iostream>

#include "RandomGenerator.h"

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