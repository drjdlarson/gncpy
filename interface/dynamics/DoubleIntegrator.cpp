#include <memory>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include <pybind11/functional.h> // needed to allow pickling of std::function
#include <gncpy/dynamics/ILinearDynamics.h>
#include <gncpy/dynamics/DoubleIntegrator.h>

#include "../Macros.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initDoubleIntegrator(py::module& m) {

    using namespace lager;

    GNCPY_PY_CHILD_CLASS(gncpy::dynamics::DoubleIntegrator, gncpy::dynamics::ILinearDynamics)(m, "DoubleIntegrator")
        .def(py::init<double>())
        GNCPY_DYNAMICS_ILINEARDYNAMICS_INTERFACE(gncpy::dynamics::DoubleIntegrator)
        .def_property("dt", &gncpy::dynamics::DoubleIntegrator::dt, &gncpy::dynamics::DoubleIntegrator::setDt) // Essentially setter and getter
        GNCPY_PICKLE(gncpy::dynamics::DoubleIntegrator);

}