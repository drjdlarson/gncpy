#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../Macros.h"

namespace py = pybind11;


void initMatrix(py::module&);


PYBIND11_MODULE(_math, m) {
    initMatrix(m);


}