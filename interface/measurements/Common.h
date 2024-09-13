#pragma once
#include <Eigen/Dense>
#include <gncpy/measurements/Parameters.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

extern void initInterface(pybind11::module&);
extern void initParameters(pybind11::module&);
extern void initStateObservation(pybind11::module&);
extern void initRangeAndBearing(pybind11::module&);

#define GNCPY_MEASUREMENTS_IMEASMODEL_INTERFACE(cName) \
    .def("get_meas_mat", pybind11::overload_cast<const Eigen::VectorXd&, const lager::gncpy::measurements::MeasParams*>(&cName::getMeasMat, pybind11::const_), \
        pybind11::arg("state"), pybind11::arg_v("params", static_cast<lager::gncpy::measurements::MeasParams *>(nullptr), "lager::gncpy::measurements::MeasParams*=nullptr")) \
    .def("measure", pybind11::overload_cast<const Eigen::VectorXd&, const lager::gncpy::measurements::MeasParams*>(&cName::measure, pybind11::const_), \
         pybind11::arg("state"), pybind11::arg_v("params", static_cast<lager::gncpy::measurements::MeasParams *>(nullptr), "lager::gncpy::measurements::MeasParams*=nullptr"))
    