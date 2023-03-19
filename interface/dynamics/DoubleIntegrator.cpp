#include <pybind11/pybind11.h>
#include <gncpy/dynamics/ILinearDynamics.h>
#include <gncpy/dynamics/DoubleIntegrator.h>
#include "../math/Common.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initDoubleIntegrator(py::module& m) {

    using namespace lager;

    py::class_<gncpy::dynamics::DoubleIntegrator<double>, gncpy::dynamics::ILinearDynamics<double>>(m, "DoubleIntegrator")
        .def(py::init<double>())
        GNCPY_DYNAMICS_ILINEARDYNAMICS_INTERFACE(gncpy::dynamics::DoubleIntegrator<double>, double)
        .def_property("dt", &gncpy::dynamics::DoubleIntegrator<double>::dt, &gncpy::dynamics::DoubleIntegrator<double>::setDt);
        
}