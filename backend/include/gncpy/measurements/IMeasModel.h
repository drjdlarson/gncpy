#pragma once
#include <functional>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/measurements/Parameters.h"



namespace lager::gncpy::measurements{
template<typename T>
class IMeasModel {
public:
    virtual matrix::Vector<T> measure(const matrix::Vector<T>& state, const std::unique_ptr<MeasParams>& params=std::make_unique<MeasParams>()) const = 0;
    virtual matrix::Matrix<T> getMeasMat(const matrix::Vector<T>& state, const std::unique_ptr<MeasParams>& params=std::make_unique<MeasParams>()) const = 0;
};
} // namespace lager::gncpy::measurements