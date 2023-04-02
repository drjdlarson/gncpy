#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include <pybind11/functional.h> // needed to allow pickling of std::function
#include <gncpy/dynamics/ILinearDynamics.h>
#include <gncpy/dynamics/DoubleIntegrator.h>
#include "../math/Common.h"
#include "../Macros.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initDoubleIntegrator(py::module& m) {

    using namespace lager;

    GNCPY_PY_CHILD_CLASS(gncpy::dynamics::DoubleIntegrator<double>, gncpy::dynamics::ILinearDynamics<double>)(m, "DoubleIntegrator")
        .def(py::init<double>())
        GNCPY_DYNAMICS_ILINEARDYNAMICS_INTERFACE(gncpy::dynamics::DoubleIntegrator<double>, double)
        .def_property("dt", &gncpy::dynamics::DoubleIntegrator<double>::dt, &gncpy::dynamics::DoubleIntegrator<double>::setDt); // Essentially setter and getter
        // .def(py::pickle(
        //     [](const gncpy::dynamics::DoubleIntegrator<double>& p) { // __getstate__
        //         return py::make_tuple(p.dt(), p.hasControlModel(), p.controlModel(), p.hasStateConstraint(), p.stateConstraints());
        //     },
        //     [](py::tuple t) { // __setstate__
        //         if(t.size() != 5) {
        //             throw std::runtime_error("Invalid state!");
        //         }

        //         gncpy::dynamics::DoubleIntegrator dynObj(t[0].cast<double>());
                
        //         if(t[1].cast<bool>()) {
        //             dynObj.setControlModel(t[2].cast<std::function<gncpy::matrix::Matrix<double> (double timestep, const gncpy::dynamics::ControlParams* controlParams)>>());
        //         }
        //         if(t[3].cast<bool>()) {
        //             dynObj.setStateConstraints(t[3].cast<std::function<void (double timestep, gncpy::matrix::Vector<double>& state, const gncpy::dynamics::ConstraintParams* const constraintParams)>>());
        //         }

        //         return dynObj;
        //     }
        // ));
        
}