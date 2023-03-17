#pragma once
#include <gncpy/dynamics/Parameters.h>
#include <gncpy/math/Vector.h>


/*
    see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python and
    https://pybind11.readthedocs.io/en/stable/advanced/classes.html#virtual-and-inheritance for methods for using the c++ classes for doing the inheritance
    in python (using the c++ versions as the base class for things like DyanmicsBase). This is not used right now because the python and c++ code take
    different values for their overridden functions (ie the interface in python is not clean). Instead a macro is defined to be used in every c++ wrapper dynamics class
    to avoid repeating common code for the interface functions. Then the python code can call the appropriate methods. This has the downside of keeping the inheritance in python and c++ decoupled
    (ie you can have a situation where in python DoubleInt -> LinearBase -> DynamicsBase but in c++ DoubleInt -> DyanmicsBase if you aren't careful)
*/
// using namespace lager;

// template<class DynBase = gncpy::dynamics::IDynamics<double>> class PyDynamics : public gncpy::dynamics::IDynamics<double> {
//     public:
//     using gncpy::dynamics::IDynamics<double>::gncpy::dynamics::IDynamics<double>;  // Inherit constructors
//     gncpy::matrix::Vector<double> propagateState(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control) const override {
//         using return_t = gncpy::matrix::Vector<double>;
//         using class_t = gncpy::dynamics::IDynamics<double>;
//         PYBIND11_OVERRIDE_PURE(
//             return_t, // return type
//             class_t, // parent class
//             propagateState, /* Name of function in C++ (must match Python name) */
//             function_args // function argumets
//         )
//     };
// };


#define GNCPY_DYNAMICS_PROPAGATE_STATE_INTERFACE(cName, T) \
    .def("propagate_state", py::overload_cast<T, const gncpy::matrix::Vector<T>&, const gncpy::dynamics::StateTransParams*>(&cName::propagateState, py::const_), \
          py::arg("timestep"), py::arg("state"), py::arg_v("stateTransParams", static_cast<gncpy::dynamics::StateTransParams *>(nullptr), "lager::gncpy::dynamics::StateTransParams*=nullptr")) \
    .def("propagate_state", py::overload_cast<T, const gncpy::matrix::Vector<T>&, const gncpy::matrix::Vector<T>&>(&cName::propagateState, py::const_)) \
    .def("propagate_state", py::overload_cast<T, const gncpy::matrix::Vector<T>&, const gncpy::matrix::Vector<T>&, const gncpy::dynamics::StateTransParams* const, const gncpy::dynamics::ControlParams* const, const gncpy::dynamics::ConstraintParams* const>(&cName::propagateState, py::const_))


#define GNCPY_DYNAMICS_IDYNAMICS_INTERFACE(cName, T) \
    .def("state_names", &cName::stateNames) \
    GNCPY_DYNAMICS_PROPAGATE_STATE_INTERFACE(cName, T)


#define GNCPY_DYNAMICS_ILINEARDYNAMICS_INTERFACE(cName, T) \
    GNCPY_DYNAMICS_IDYNAMICS_INTERFACE(cName, T) \
    .def("get_state_mat", &cName::getStateMat, \
         py::arg("timestep"), py::arg_v("stateTransParams", static_cast<gncpy::dynamics::StateTransParams *>(nullptr), "lager::gncpy::dynamics::StateTransParams*=nullptr")) \
    .def("get_input_mat", &cName::getInputMat, \
         py::arg("timestep"), py::arg_v("controlParams", static_cast<gncpy::dynamics::ControlParams *>(nullptr), "lager::gncpy::dynamics::ControlParams*=nullptr"))