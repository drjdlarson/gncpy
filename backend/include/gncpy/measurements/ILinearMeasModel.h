#pragma once
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/measurements/Parameters.h"
#include "gncpy/measurements/IMeasModel.h"

namespace lager::gncpy::measurements
{
template<typename T>
class ILinearMeasModel : public IMeasModel<T> {
public:
    matrix::Vector<T> measure(const matrix::Vector<T>& state, const MeasParams* params=nullptr) const override {
        return this->getMeasMat(state, params) * state;
    }
};
} // namespace lager::gncpy::measurements
