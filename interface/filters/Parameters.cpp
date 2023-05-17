#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include <gncpy/dynamics/Parameters.h>
#include <gncpy/filters/Parameters.h>

#include "Common.h"
#include "../Macros.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initParameters(py::module& m) {

    using namespace lager;

    GNCPY_PY_BASE_CLASS(gncpy::filters::BayesPredictParams)(m, "BayesPredictParams")
        .def(py::init())
        .def_readwrite("stateTransParams", &gncpy::filters::BayesPredictParams::stateTransParams)
        .def_readwrite("controlParams", &gncpy::filters::BayesPredictParams::controlParams)
        GNCPY_PICKLE(gncpy::filters::BayesPredictParams);
    
    GNCPY_PY_BASE_CLASS(gncpy::filters::BayesCorrectParams)(m, "BayesCorrectParams")
        .def(py::init())
        .def_readwrite("measParams", &gncpy::filters::BayesCorrectParams::measParams)
        GNCPY_PICKLE(gncpy::filters::BayesCorrectParams);
}