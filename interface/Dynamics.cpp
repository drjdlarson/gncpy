#include <memory>
#include <pybind11/pybind11.h>
#include <gncpy/dynamics/DoubleIntegrator.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace lager;

// see https://stackoverflow.com/questions/44925081/how-to-wrap-templated-classes-with-pybind11 for pybind with template class
PYBIND11_MODULE(_dynamics, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: scikit_build_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    py::class_<gncpy::dynamics::DoubleIntegrator<double>, std::shared_ptr<gncpy::dynamics::DoubleIntegrator<double>>, gncpy::dynamics::LinearDynamics<double>>(m, "DoubleIntegrator")
        .def(py::init<double>());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}