/**
 * @file pisinger.cpp
 * @author Alejandro Marrero (amarrerd@ull.edu.es)
 * @brief
 * @version 0.1
 * @date 2024-05-29
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <pybind11/pybind11.h>

#include "pybind11/numpy.h"
namespace py = pybind11;

void init_combo(py::module_ &);
void init_minknap(py::module_ &);
void init_expknap(py::module_ &);

PYBIND11_MODULE(pisinger_cpp, m) {
    m.doc() = "Pybinding for the Pisinger Solvers";

    init_minknap(m);
    init_combo(m);
    init_expknap(m);
}