#include <pybind11/pybind11.h>
#include <gncpy/measurements/RangeAndBearing.h>
#include <gncpy/measurements/Parameters.h>
#include "../math/Common.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initRangeAndBearing(py::module& m) {
    using namespace lager;

    py::class_<gncpy::measurements::RangeAndBearingParams, gncpy::measurements::MeasParams>(m, "RangeAndBearingParams")
        .def(py::init<uint8_t, uint8_t>())
        .def_readwrite("x_ind", &gncpy::measurements::RangeAndBearingParams::xInd)
        .def_readwrite("y_ind", &gncpy::measurements::RangeAndBearingParams::yInd);
    
    py::class_<gncpy::measurements::RangeAndBearing<double>, gncpy::measurements::INonLinearMeasModel<double>>(m, "RangeAndBearing")
        .def(py::init())
        GNCPY_MEASUREMENTS_IMEASMODEL_INTERFACE(gncpy::measurements::RangeAndBearing<double>, double);
}