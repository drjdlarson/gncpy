#include <memory>
#include <pybind11/pybind11.h>
#include <gncpy/measurements/IMeasModel.h>
#include <gncpy/measurements/ILinearMeasModel.h>
#include <gncpy/measurements/INonLinearMeasModel.h>
#include "../Macros.h"


namespace py = pybind11;


// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initInterface(py::module& m) {
    using namespace lager;

    // define these so the inherited classes import ok
    GNCPY_PY_BASE_CLASS(gncpy::measurements::IMeasModel<double>)(m, "IMeasModel");
    GNCPY_PY_CHILD_CLASS(gncpy::measurements::ILinearMeasModel<double>, gncpy::measurements::IMeasModel<double>)(m, "ILinearMeasModel");
    GNCPY_PY_CHILD_CLASS(gncpy::measurements::INonLinearMeasModel<double>, gncpy::measurements::IMeasModel<double>)(m, "INonLinearMeasModel");
}