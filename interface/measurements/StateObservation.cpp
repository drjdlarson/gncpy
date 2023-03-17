#include <pybind11/pybind11.h>
#include <gncpy/measurements/StateObservation.h>
#include "../math/Common.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initStateObservation(py::module& m) {

    using namespace lager;

    py::class_<gncpy::measurements::StateObservation<double>, gncpy::measurements::ILinearMeasModel<double>>(m, "StateObservation")
        .def(py::init<double>())
        .def("measure", &gncpy::measurements::StateObservation<double>::measure,
             py::arg("state"), py::arg_v("params", static_cast<gncpy::measurements::MeasParams *>(nullptr), "lager::gncpy::measurements::MeasParams*=nullptr"))
        .def("get_meas_mat", &gncpy::measurements::StateObservation<double>::getMeasMat,
            py::arg("state"), py::arg_v("params", static_cast<gncpy::measurements::MeasParams *>(nullptr), "lager::gncpy::measurements::MeasParams*=nullptr"));

}