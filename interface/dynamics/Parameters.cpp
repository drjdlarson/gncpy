#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include <gncpy/dynamics/Parameters.h>
#include "../Macros.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initParameters(py::module& m) {

    using namespace lager;

    // see https://github.com/pybind/pybind11/issues/956 for why the shared_ptr is needed
    GNCPY_PY_BASE_CLASS(gncpy::dynamics::StateTransParams)(m, "StateTransParams")
        .def(py::init());
        // .def(py::pickle(
        //     []([[maybe_unused]] const gncpy::dynamics::StateTransParams& p) { // __getstate__
        //         return py::make_tuple();
        //     },
        //     []([[maybe_unused]] py::tuple t) { // __setstate__
        //         return gncpy::dynamics::StateTransParams();
        //     }
        // ));
    
    GNCPY_PY_BASE_CLASS(gncpy::dynamics::ControlParams)(m, "ControlParams")
        .def(py::init());
        // .def(py::pickle(
        //     []([[maybe_unused]] const gncpy::dynamics::ControlParams& p) { // __getstate__
        //         return py::make_tuple();
        //     },
        //     []([[maybe_unused]] py::tuple t) { // __setstate__
        //         return gncpy::dynamics::ControlParams();
        //     }
        // ));

    GNCPY_PY_BASE_CLASS(gncpy::dynamics::ConstraintParams)(m, "ConstraintParams")
        .def(py::init())
        .def(py::pickle(
            []([[maybe_unused]] const gncpy::dynamics::ConstraintParams& p) { // __getstate__
                return py::make_tuple();
            },
            []([[maybe_unused]]  py::tuple t) { // __setstate__
                return gncpy::dynamics::ConstraintParams();
            }
        ));
}