#pragma once
#include <pybind11/pybind11.h>
#include <gncpy/measurements/Parameters.h>
#include <gncpy/math/Vector.h>

#define GNCPY_MEASUREMENTS_IMEASMODEL_INTERFACE(cName, T) \
    .def("get_meas_mat", py::overload_cast<const gncpy::matrix::Vector<T>&, const gncpy::measurements::MeasParams*>(&cName::getMeasMat, py::const_), \
        py::arg("state"), py::arg_v("params", static_cast<gncpy::measurements::MeasParams *>(nullptr), "lager::gncpy::measurements::MeasParams*=nullptr")) \
    .def("measure", py::overload_cast<const gncpy::matrix::Vector<T>&, const gncpy::measurements::MeasParams*>(&cName::measure, py::const_), \
         py::arg("state"), py::arg_v("params", static_cast<gncpy::measurements::MeasParams *>(nullptr), "lager::gncpy::measurements::MeasParams*=nullptr"))
    