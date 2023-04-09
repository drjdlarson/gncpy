#include <memory>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // needed because some backend gncpy functions retrun stl types
#include <pybind11/functional.h> // needed to allow pickling of std::function
#include <gncpy/dynamics/ILinearDynamics.h>
#include <gncpy/dynamics/DoubleIntegrator.h>

#include "../math/Common.h"
#include "../Macros.h"
#include "Common.h"

namespace py = pybind11;

// see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-classes-with-template-parameters for pybind with template class
// see https://stackoverflow.com/questions/62854830/error-by-wrapping-c-abstract-class-with-pybind11 for ImportError: generic_type: type "Derived" referenced unknown base type "Base" error
void initDoubleIntegrator(py::module& m) {

    using namespace lager;

    GNCPY_PY_CHILD_CLASS(gncpy::dynamics::DoubleIntegrator<double>, gncpy::dynamics::ILinearDynamics<double>)(m, "DoubleIntegrator")
        .def(py::init<double>())
        GNCPY_DYNAMICS_ILINEARDYNAMICS_INTERFACE(gncpy::dynamics::DoubleIntegrator<double>, double)
        .def_property("dt", &gncpy::dynamics::DoubleIntegrator<double>::dt, &gncpy::dynamics::DoubleIntegrator<double>::setDt) // Essentially setter and getter
        .def(py::pickle(
            [](gncpy::dynamics::DoubleIntegrator<double>& p) { // __getstate__
                std::string ssb = p.saveClassState().str();
                return py::make_tuple(py::bytes(ssb.str()));
            },
            [](py::tuple t) { // __setstate__
                if(t.size() != 1) {
                    throw std::runtime_error("Invalid state!");
                }

                std::stringstream ssb(t[0].cast<std::string>(), std::ios::in | std::ios::out | std::ios::binary);
                auto dynObj = lager::gncpy::dynamics::DoubleIntegrator<double>::loadClass(ssb);

                return dynObj;
            }
        ))
        .def("__str__", &gncpy::dynamics::DoubleIntegrator<double>::toJSON)
        .def("__repr__", &gncpy::dynamics::DoubleIntegrator<double>::toJSON);

}