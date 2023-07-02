#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <gncpy/Exceptions.h>
#include <gncpy/measurements/RangeAndBearing.h>
#include <gncpy/measurements/Parameters.h>
#include "../Macros.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initRangeAndBearing(py::module& m) {
    using namespace lager;

    GNCPY_PY_CHILD_CLASS(gncpy::measurements::RangeAndBearingParams, gncpy::measurements::MeasParams)(m, "RangeAndBearingParams")
        .def(py::init<uint8_t, uint8_t>())
        .def_readwrite("x_ind", &gncpy::measurements::RangeAndBearingParams::xInd)
        .def_readwrite("y_ind", &gncpy::measurements::RangeAndBearingParams::yInd)
        GNCPY_PICKLE(gncpy::measurements::RangeAndBearingParams);
    
    GNCPY_PY_CHILD_CLASS(gncpy::measurements::RangeAndBearing, gncpy::measurements::INonLinearMeasModel)(m, "RangeAndBearing")
        .def(py::init())
        GNCPY_MEASUREMENTS_IMEASMODEL_INTERFACE(gncpy::measurements::RangeAndBearing)
        .def("args_to_params", []([[maybe_unused]] gncpy::measurements::RangeAndBearing& self, py::tuple args) {
            if(args.size() != 2){
                throw gncpy::exceptions::BadParams("Must pass x and y indices to the range and bearing model");
            }
            std::uint8_t xind = py::cast<uint8_t>(args[0]);
            std::uint8_t yind = py::cast<uint8_t>(args[1]);
            return gncpy::measurements::RangeAndBearingParams(xind, yind);
        })
        GNCPY_PICKLE(gncpy::measurements::RangeAndBearing);
}