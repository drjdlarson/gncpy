#pragma once
#include "Matrix.h"

namespace lager::gncpy::dynamics {

template<typename T>
class IDynamics {
public:
    virtual matrix::Matrix<T> propagateState(T timestep, const matrix::Matrix<T>& state) const = 0;
    virtual matrix::Matrix<T> getStateMat(T timestep) const = 0;
    virtual matrix::Matrix<T> getInputMat(T timestep) const = 0;
};
    
} // namespace lager::gncpy::dynamics 