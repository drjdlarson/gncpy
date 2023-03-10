#pragma once
#include "gncpy/dynamics/IDynamics.h"
#include "gncpy/dynamics/Exceptions.h"

namespace lager::gncpy::dynamics {

template<typename T>
class ILinearDynamics : public IDynamics<T> {
public:
    matrix::Matrix<T> propagateState(T timestep, const matrix::Matrix<T>& state, const matrix::Matrix<T>& control) const override {
        matrix::Matrix<T> nextState = this->getStateMat(timestep) * state;

        if(this->hasControlModel()){
            nextState += this->getInputMat(timestep, nextState, control) * control;
        }

        if(this->hasStateConstraint()){
            this->stateConstraint(timestep, nextState);
        }

        return nextState;
    }

    matrix::Matrix<T> getInputMat(T timestep, const matrix::Matrix<T>& state, const matrix::Matrix<T>& control) const override{
        return this->controlModel(timestep, state, control);
    }

};
    
} // namespace lager::gncpy::dynamics
