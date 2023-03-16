#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include "../Macros.h"


namespace py = pybind11;

void initInterface(py::module&);
void initParameters(py::module&);
void initDoubleIntegrator(py::module&);


PYBIND11_MODULE(_dynamics, m) {

    initInterface(m);
    initParameters(m);

    initDoubleIntegrator(m);


    #ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = "dev";
    #endif
}