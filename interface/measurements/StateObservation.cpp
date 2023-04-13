#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gncpy/Exceptions.h>
#include <gncpy/measurements/StateObservation.h>
#include <gncpy/measurements/Parameters.h>
#include "../Macros.h"
#include "../math/Common.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initStateObservation(py::module& m) {

    using namespace lager;

    GNCPY_PY_CHILD_CLASS(gncpy::measurements::StateObservationParams, gncpy::measurements::MeasParams)(m, "StateObservationParams")
        .def(py::init<const std::vector<uint8_t>&>())
        .def_readonly("obs_inds", &gncpy::measurements::StateObservationParams::obsInds, "Indices of the state vector to measure (read-only)")
        GNCPY_PICKLE(gncpy::measurements::StateObservationParams);

    GNCPY_PY_CHILD_CLASS(gncpy::measurements::StateObservation<double>, gncpy::measurements::ILinearMeasModel<double>)(m, "StateObservation")
        .def(py::init())
        GNCPY_MEASUREMENTS_IMEASMODEL_INTERFACE(gncpy::measurements::StateObservation<double>, double)
        .def("args_to_params", []([[maybe_unused]] gncpy::measurements::StateObservation<double>& self, py::tuple args) {
            if(args.size() != 1){
                throw gncpy::exceptions::BadParams("Must only pass indices to state observation model");
            }
            std::vector<uint8_t> inds;
            for(auto& ii : args[0]) { // only element in args is a list of indices
                inds.emplace_back(py::cast<uint8_t>(ii));
            }
            // return std::make_shared<gncpy::measurements::StateObservationParams>(inds);
            return gncpy::measurements::StateObservationParams(inds);
        })
        GNCPY_PICKLE(gncpy::measurements::StateObservation<double>);
}