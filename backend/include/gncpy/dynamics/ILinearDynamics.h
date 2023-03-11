#pragma once
#include "gncpy/dynamics/IDynamics.h"
#include "gncpy/math/Vector.h"

namespace lager::gncpy::dynamics {

template<typename T>
class LinearDynamics : public IDynamics<T> {
public:
    matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control) const override {
        matrix::Vector<T> nextState = this->getStateMat(timestep) * state;

        if(this->hasControlModel()){
            nextState += this->getInputMat(timestep, nextState, control) * control;
        }

        if(this->hasStateConstraint()){
            this->stateConstraint(timestep, nextState);
        }

        return nextState;
    }

    matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state) const override {
        return this->propagateState(timestep, state, matrix::Vector<T>(state.numRows()));
    }

    matrix::Matrix<T> getInputMat(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control) const override{
        return this->controlModel(timestep, state, control);
    }

private:
};
    
} // namespace lager::gncpy::dynamics
