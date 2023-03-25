#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gncpy/measurements/RangeAndBearing.h>
#include <gncpy/measurements/Parameters.h>
#include "../Macros.h"
#include "../math/Common.h"
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
        .def(py::pickle(
            []([[maybe_unused]] const gncpy::measurements::RangeAndBearingParams& p) { // __getstate__
                return py::make_tuple(p.xInd, p.yInd);
            },
            []([[maybe_unused]] py::tuple t) { // __setstate__
                if(t.size() != 2){
                    throw std::runtime_error("Invalid state!");
                }
                return gncpy::measurements::RangeAndBearingParams(t[0].cast<uint8_t>(), t[1].cast<uint8_t>());
            }
        ));
    
    GNCPY_PY_CHILD_CLASS(gncpy::measurements::RangeAndBearing<double>, gncpy::measurements::INonLinearMeasModel<double>)(m, "RangeAndBearing")
        .def(py::init())
        GNCPY_MEASUREMENTS_IMEASMODEL_INTERFACE(gncpy::measurements::RangeAndBearing<double>, double);
}