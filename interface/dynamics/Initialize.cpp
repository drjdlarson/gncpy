#include <pybind11/pybind11.h>
#include "Common.h"
#include "../Macros.h"


PYBIND11_MODULE(_dynamics, m) {

    initInterface(m);
    initParameters(m);

    initDoubleIntegrator(m);
    initClohessyWiltshire2D(m);
    initClohessyWiltshire(m);


    #ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = "dev";
    #endif
}