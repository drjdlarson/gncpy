#include <pybind11/pybind11.h>
#include <gncpy/dynamics/IDynamics.h>
#include <gncpy/dynamics/ILinearDynamics.h>

namespace py = pybind11;


// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initInterface(py::module& m) {
    using namespace lager::gncpy;

    // define these so the inherited classes import ok
    py::class_<dynamics::IDynamics<double>>(m, "IDynamics");
    py::class_<dynamics::ILinearDynamics<double>, dynamics::IDynamics<double>>(m, "ILinearDynamics");
}