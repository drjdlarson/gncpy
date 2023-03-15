#pragma once
#include <functional>
#include <vector>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Math.h"
#include "gncpy/measurements/Parameters.h"
#include "gncpy/measurements/IMeasModel.h"

namespace lager::gncpy::measurements
{
template<typename T>
class ILinearMeasModel : public IMeasModel<T> {
public:
    matrix::Vector<T> measure(const matrix::Vector<T>& state, const std::unique_ptr<MeasParams>&params=std::make_unique<MeasParams>()) const override {
        matrix::Matrix measMat = this->getMeasMat(state, params);
        return measMat * state;
    }
};
} // namespace lager::gncpy::measurements
