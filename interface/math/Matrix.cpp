#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <gncpy/math/Matrix.h>
#include "../Macros.h"
#include "Common.h"

namespace py = pybind11;


PYBIND11_MODULE(_matrix, m) {

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}