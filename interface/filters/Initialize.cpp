#include <pybind11/pybind11.h>
#include "../Macros.h"


namespace py = pybind11;

void initInterface(py::module&);
void initParameters(py::module&);
void initKalman(py::module&);


PYBIND11_MODULE(_filters, m) {
    initInterface(m);
    initParameters(m);

    initKalman(m);

    #ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = "dev";
    #endif
}