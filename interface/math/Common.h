#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <gncpy/math/Matrix.h>
#include <gncpy/math/Vector.h>


namespace py = pybind11;


// Note this is in a header because it needs to be included by every translation unit (think cpp/object file) that uses matrix, so the header can be included
// see https://stackoverflow.com/questions/42645228/cast-numpy-array-to-from-custom-c-matrix-class-using-pybind11 
// and https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html for details
namespace PYBIND11_NAMESPACE {
    namespace detail {
        // copies on python -> c++ but not c++ -> python
        template<typename T>
        struct type_caster<lager::gncpy::matrix::Matrix<T>> {
        public:
            PYBIND11_TYPE_CASTER(lager::gncpy::matrix::Matrix<T>, const_name("lager::gncpy::matrix::Matrix<T>"));

            // conversion from python -> c++
            bool load(py::handle src, bool convert) {
                if(!convert && !py::array_t<T>::check_(src)) {
                    return false;
                }

                auto buf = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
                if(!buf) {
                    return false;
                }

                auto dims = buf.ndim();
                if(dims < 1 || dims > 2) {
                    return false;
                }

                std::vector<size_t> shape({buf.shape()[0], buf.shape()[1]});
                value = lager::gncpy::matrix::Matrix<T>(shape, buf.data());
                return true;
            }

            // conversion from c++ -> python
            static py::handle cast(const lager::gncpy::matrix::Matrix<T>& src, [[maybe_unused]] py::return_value_policy policy, py::handle parent) {
                py::array a(std::move(src.shape()), std::move(src.strides(true)), src.data(), parent);
                return a.release();
            }
        };


        // copies on python -> c++ but not c++ -> python
        template<typename T>
        struct type_caster<lager::gncpy::matrix::Vector<T>> {
        public:
            PYBIND11_TYPE_CASTER(lager::gncpy::matrix::Vector<T>, const_name("lager::gncpy::matrix::Vector<T>"));

            // conversion from python -> c++
            bool load(py::handle src, bool convert) {
                if(!convert && !py::array_t<T>::check_(src)) {
                    return false;
                }

                auto buf = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
                if(!buf) {
                    return false;
                }

                auto dims = buf.ndim();
                if(dims < 1 || dims > 2) {
                    return false;
                }

                std::vector<size_t> shape;
                if(dims == 1) {
                    shape.emplace_back(buf.shape()[0]);
                    shape.emplace_back(1);
                } else {
                    shape.emplace_back(buf.shape()[0]);
                    shape.emplace_back(buf.shape()[1]);
                }
                value = lager::gncpy::matrix::Vector<T>(shape, buf.data());
                return true;
            }

            // conversion from c++ -> python
            static py::handle cast(const lager::gncpy::matrix::Vector<T>& src, [[maybe_unused]] py::return_value_policy policy, py::handle parent) {
                py::array a(std::move(src.shape()), std::move(src.strides(true)), src.data(), parent);
                return a.release();
            }
        };

    }
}