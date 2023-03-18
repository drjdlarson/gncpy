#include <pybind11/pybind11.h>
#include <gncpy/measurements/IMeasModel.h>
#include <gncpy/measurements/ILinearMeasModel.h>
#include <gncpy/measurements/INonLinearMeasModel.h>


namespace py = pybind11;


// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initInterface(py::module& m) {
    using namespace lager;

    // define these so the inherited classes import ok
    py::class_<gncpy::measurements::IMeasModel<double>>(m, "IMeasModel");
    py::class_<gncpy::measurements::ILinearMeasModel<double>, gncpy::measurements::IMeasModel<double>>(m, "ILinearMeasModel");
    py::class_<gncpy::measurements::INonLinearMeasModel<double>, gncpy::measurements::IMeasModel<double>>(m, "INonLinearMeasModel");
}