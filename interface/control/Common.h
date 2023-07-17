#pragma once
#include <Eigen/Dense>
#include <gncpy/control/Parameters.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

void initInterface(pybind11::module&);
void initParameters(pybind11::module&);
void initStateControl(pybind11::module&);

#define GNCPY_CONTROL_ICONTROLMODEL_INTERFACE(cName) \
    .def("get_input_mat", pybind11::overload_cast<const Eigen::VectorXd&, const lager::gncpy::control::ControlParams*>(&cName::getInputMat, pybind11::const_), \
        pybind11::arg("state"), pybind11::arg_v("params", static_cast<lager::gncpy::control::ControlParams *>(nullptr), "lager::gncpy::control::ControlParams*=nullptr")) \
    .def("get_control_inputs", pybind11::overload_cast<const Eigen::VectorXd&, const Eigen::VectorXd&, const lager::gncpy::control::ControlParams*>(&cName::getControlInput, pybind11::const_), \
         pybind11::arg("state"), pybind11::arg("input"), pybind11::arg_v("params", static_cast<lager::gncpy::control::ControlParams *>(nullptr), "lager::gncpy::control::ControlParams*=nullptr"))
    