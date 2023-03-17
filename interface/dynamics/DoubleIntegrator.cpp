#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include <gncpy/dynamics/IDynamics.h>
#include <gncpy/dynamics/ILinearDynamics.h>
#include <gncpy/dynamics/DoubleIntegrator.h>
#include "../math/Matrix.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initDoubleIntegrator(py::module& m) {

    using namespace lager::gncpy;

    py::class_<dynamics::DoubleIntegrator<double>, dynamics::ILinearDynamics<double>>(m, "DoubleIntegrator")
        .def(py::init<double>())
        .def("state_names", &dynamics::DoubleIntegrator<double>::stateNames)
        .def("get_state_mat", &dynamics::DoubleIntegrator<double>::getStateMat,
                py::arg("timestep"), py::arg_v("stateTransParams", static_cast<dynamics::StateTransParams *>(nullptr), "lager::gncpy::dynamics::StateTransParams*=nullptr"))
        .def_property("dt", &dynamics::DoubleIntegrator<double>::dt, &dynamics::DoubleIntegrator<double>::setDt);
        
}