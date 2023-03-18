#include <pybind11/pybind11.h>
#include "../Macros.h"


namespace py = pybind11;

void initInterface(py::module&);
void initParameters(py::module&);
void initStateObservation(py::module& m);
void initRangeAndBearing(py::module& m);


PYBIND11_MODULE(_measurements, m) {

    initInterface(m);
    initParameters(m);

    initStateObservation(m);
    initRangeAndBearing(m);

    #ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = "dev";
    #endif
}