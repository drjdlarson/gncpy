#pragma once
#include "dynamics/IDynamics.h"

namespace lager::gncpy::dynamics {

template<typename T>
class ILinearDynamics : IDynamics<T> {
public:
    matrix::Matrix<T> propagateState(T timestep, const matrix::Matrix<T>& state) const override {
        matrix::Matrix<T> nextState = getStateMat(timestep) * state;

        // TODO: implement control model changes

        return state;
    }

};
    
} // namespace lager::gncpy::dynamics
