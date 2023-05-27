#pragma once
#include <Eigen/Dense>
#include <gncpy/dynamics/Parameters.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


extern void initInterface(pybind11::module&);
extern void initParameters(pybind11::module&);
extern void initDoubleIntegrator(pybind11::module&);


#define GNCPY_DYNAMICS_PROPAGATE_STATE_INTERFACE(cName) \
    .def("propagate_state", pybind11::overload_cast<double, const Eigen::VectorXd&, const lager::gncpy::dynamics::StateTransParams*>(&cName::propagateState, pybind11::const_), \
          pybind11::arg("timestep"), pybind11::arg("state"), pybind11::arg_v("stateTransParams", static_cast<lager::gncpy::dynamics::StateTransParams *>(nullptr), "lager::gncpy::dynamics::StateTransParams*=nullptr")) \
    .def("propagate_state", pybind11::overload_cast<double, const Eigen::VectorXd&, const Eigen::VectorXd&>(&cName::propagateState, pybind11::const_)) \
    .def("propagate_state", pybind11::overload_cast<double, const Eigen::VectorXd&, const Eigen::VectorXd&, const lager::gncpy::dynamics::StateTransParams* const, const lager::gncpy::dynamics::ControlParams* const, const lager::gncpy::dynamics::ConstraintParams* const>(&cName::propagateState, pybind11::const_))


#define GNCPY_DYNAMICS_IDYNAMICS_INTERFACE(cName) \
    .def("state_names", &cName::stateNames) \
    GNCPY_DYNAMICS_PROPAGATE_STATE_INTERFACE(cName)


#define GNCPY_DYNAMICS_ILINEARDYNAMICS_INTERFACE(cName) \
    GNCPY_DYNAMICS_IDYNAMICS_INTERFACE(cName) \
    .def("get_state_mat", &cName::getStateMat, \
         pybind11::arg("timestep"), pybind11::arg_v("stateTransParams", static_cast<lager::gncpy::dynamics::StateTransParams *>(nullptr), "lager::gncpy::dynamics::StateTransParams*=nullptr")) \
    .def("get_input_mat", &cName::getInputMat, \
         pybind11::arg("timestep"), pybind11::arg_v("controlParams", static_cast<lager::gncpy::dynamics::ControlParams *>(nullptr), "lager::gncpy::dynamics::ControlParams*=nullptr"))
