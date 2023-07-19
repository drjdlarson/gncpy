#include <pybind11/pybind11.h>
#include "Common.h"
#include "../Macros.h"


PYBIND11_MODULE(_control, m) {

    initInterface(m);
    initParameters(m);

    initStateControl(m);

    #ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = "dev";
    #endif
}