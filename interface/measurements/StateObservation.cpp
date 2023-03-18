#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gncpy/measurements/StateObservation.h>
#include <gncpy/measurements/Parameters.h>
#include "../math/Common.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initStateObservation(py::module& m) {

    using namespace lager;

    py::class_<gncpy::measurements::StateObservationParams, gncpy::measurements::MeasParams>(m, "StateObservationParams")
        .def(py::init<const std::vector<uint8_t>&>())
        .def_readonly("obs_inds", &gncpy::measurements::StateObservationParams::obsInds,
        "Indices of the state vector to measure (read-only)");

    py::class_<gncpy::measurements::StateObservation<double>, gncpy::measurements::ILinearMeasModel<double>>(m, "StateObservation")
        .def(py::init())
        GNCPY_MEASUREMENTS_IMEASMODEL_INTERFACE(gncpy::measurements::StateObservation<double>, double);

}