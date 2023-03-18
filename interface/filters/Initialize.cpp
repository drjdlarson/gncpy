#include <pybind11/pybind11.h>
#include "../Macros.h"


namespace py = pybind11;

void initInterface(py::module&);
void initParameters(py::module&);
void initKalmanFilter(py::module&);


PYBIND11_MODULE(_filters, m) {
    initInterface(m);
    // initParameters(m);

    // initKalmanFilter(m);

    #ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = "dev";
    #endif
}