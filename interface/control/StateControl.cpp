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
        .def(py::init<const std::vector<uint8_t>&, const std::vector<uint8_t>&>())
        .def(py::init<const std::vector<uint8_t>&, const std::vector<uint8_t>&, const std::vector<double>&>())
        .def_readonly("cont_inds", &gncpy::control::StateControlParams::contInds, "Indices of the state vector to control (read-only)")
        GNCPY_PICKLE(gncpy::control::StateControlParams);

    GNCPY_PY_CHILD_CLASS(gncpy::control::StateControl, gncpy::control::ILinearControlModel)(m, "StateControl")
        .def(py::init<size_t, size_t>())
        GNCPY_CONTROL_ILINEARCONTROLMODEL_INTERFACE(gncpy::control::StateControl)
        .def("args_to_params", []([[maybe_unused]] gncpy::control::StateControl& self, py::tuple args) {
            if(args.size() != 3 && args.size() != 2){
                throw gncpy::exceptions::BadParams("Must only pass indices to state control model");
            }
            std::vector<uint8_t> rows;
            std::vector<uint8_t> cols;
            for(auto& ii : args[0]) { // only element in args is a list of indices
                rows.emplace_back(py::cast<uint8_t>(ii));
            }
            for(auto& ii : args[1]) { // only element in args is a list of indices
                cols.emplace_back(py::cast<uint8_t>(ii));
            }
            if (args.size() == 2) {
                return gncpy::control::StateControlParams(rows, cols);
            }
            std::vector<double> vals;
            for(auto& ii : args[2]) { // only element in args is a list of indices
                vals.emplace_back(py::cast<double>(ii));
            }
            return gncpy::control::StateControlParams(rows, cols, vals);
            
        })
        GNCPY_PICKLE(gncpy::control::StateControl);
}