#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gncpy/Exceptions.h>
#include <gncpy/control/StateControl.h>
#include <gncpy/control/Parameters.h>

#include "../Macros.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initStateControl(py::module& m) {

    using namespace lager;

    GNCPY_PY_CHILD_CLASS(gncpy::control::StateControlParams, gncpy::control::ControlParams) (m, "StateControlParams")
        .def(py::init<const std::vector<uint8_t>&>())
        .def_readonly("cont_inds", &gncpy::control::StateControlParams::contInds, "Indices of the state vector to control (read-only)")
        GNCPY_PICKLE(gncpy::control::StateControlParams);

    GNCPY_PY_CHILD_CLASS(gncpy::control::StateControl, gncpy::control::ILinearControlModel)(m, "StateControl")
        .def(py::init())
        GNCPY_CONTROL_ICONTROLMODEL_INTERFACE(gncpy::control::StateControl)
        .def("args_to_params", []([[maybe_unused]] gncpy::control::StateControl& self, py::tuple args) {
            if(args.size() != 1){
                throw gncpy::exceptions::BadParams("Must only pass indices to state control model");
            }
            std::vector<uint8_t> inds;
            for(auto& ii : args[0]) { // only element in args is a list of indices
                inds.emplace_back(py::cast<uint8_t>(ii));
            }
            return gncpy::control::StateControlParams(inds);
        })
        GNCPY_PICKLE(gncpy::control::StateControl);
}