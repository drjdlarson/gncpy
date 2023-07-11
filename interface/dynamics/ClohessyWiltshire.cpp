#include <memory>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include <pybind11/functional.h> // needed to allow pickling of std::function
#include <gncpy/dynamics/ILinearDynamics.h>
#include <gncpy/dynamics/ClohessyWiltshire2D.h>
#include <gncpy/dynamics/ClohessyWiltshire.h>

#include "../Macros.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initClohessyWiltshire(py::module& m) {

    using namespace lager;

    GNCPY_PY_CHILD_CLASS(gncpy::dynamics::ClohessyWiltshire, gncpy::dynamics::ClohessyWiltshire2D)(m, "ClohessyWiltshire")
        .def(py::init<double, double>())
        GNCPY_DYNAMICS_ILINEARDYNAMICS_INTERFACE(gncpy::dynamics::ClohessyWiltshire)
        .def_property("dt", &gncpy::dynamics::ClohessyWiltshire::dt, &gncpy::dynamics::ClohessyWiltshire::setDt) // Essentially setter and getter
        .def_property("mean_motion", &gncpy::dynamics::ClohessyWiltshire::mean_motion, &gncpy::dynamics::ClohessyWiltshire::setMeanMotion) // Essentially setter and getter
        GNCPY_PICKLE(gncpy::dynamics::ClohessyWiltshire);

}