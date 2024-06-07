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
