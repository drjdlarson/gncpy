#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include <gncpy/filters/IBayesFilter.h>
#include <gncpy/filters/Kalman.h>
#include "../Macros.h"
#include "../math/Common.h" // needs to be included so numpy to matrix/vector types work
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initKalman(py::module& m) {

    using namespace lager;

    GNCPY_PY_CHILD_CLASS(gncpy::filters::Kalman<double>, gncpy::filters::IBayesFilter<double>)(m, "Kalman")
        .def(py::init())
        .def("set_state_model", &gncpy::filters::Kalman<double>::setStateModel)
        .def("set_measurement_model", &gncpy::filters::Kalman<double>::setMeasurementModel)
        .def("predict", &gncpy::filters::Kalman<double>::predict)
        .def("correct", &gncpy::filters::Kalman<double>::correct)
        .def_property("cov", &gncpy::filters::Kalman<double>::covariance, &gncpy::filters::Kalman<double>::setCovariance);
}