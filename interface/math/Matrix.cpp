#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gncpy/math/Matrix.h>
#include "../Macros.h"


namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    using namespace lager::gncpy;

    py::class_<matrix::Matrix<double>>(m, "Matrix")
        .def(py::init<uint8_t, uint8_t, std::vector<double>>())
        .def(py::init<std::initializer_list<std::initializer_list<double>>>())
        .def(py::init<uint8_t, uint8_t>())
        .def("__repr__", [](const matrix::Matrix<double>& self) {
            std::ostringstream ss;
            ss << self;
            return ss.str();
        }); // see https://stackoverflow.com/questions/62415760/pybind11-how-to-invoke-the-repr-of-an-object-within-a-stdvector for details

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}