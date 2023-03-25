#pragma once
#include <functional>
#include "gncpy/dynamics/Parameters.h"
#include "gncpy/dynamics/IDynamics.h"
#include "gncpy/math/Vector.h"


namespace lager::gncpy::dynamics {

template<typename T>
class ILinearDynamics : public IDynamics<T> {
public:
    virtual matrix::Matrix<T> getStateMat(T timestep, const StateTransParams* stateTransParams=nullptr) const = 0;

    matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const StateTransParams* stateTransParams=nullptr) const override {
        matrix::Vector<T> nextState = this->propagateState_(timestep, state, stateTransParams);

        if(this->hasStateConstraint()){
            this->stateConstraint(timestep, nextState);
        }

        return nextState;
    }

    matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control) const override {
        matrix::Vector<T> nextState = this->propagateState_(timestep, state);

        if(this->hasControlModel()){
            nextState += this->getInputMat(timestep) * control;
        }

        if(this->hasStateConstraint()){
            this->stateConstraint(timestep, nextState);
        }

        return nextState;
    }

    matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control, const StateTransParams* stateTransParams, const ControlParams* controlParams, const ConstraintParams* constraintParams) const final {
        matrix::Vector<T> nextState = this->propagateState_(timestep, state, stateTransParams);

        if(this->hasControlModel()){
            nextState += this->getInputMat(timestep, controlParams) * control;
        }

        if(this->hasStateConstraint()){
            this->stateConstraint(timestep, nextState, constraintParams);
        }

        return nextState;
    }

    matrix::Matrix<T> getInputMat(T timestep, const ControlParams* controlParams=nullptr) const {
        return controlParams == nullptr ? this->controlModel(timestep) : this->controlModel(timestep, controlParams);
    }

    template<typename F>
    inline void setControlModel(F&& model) { 
        m_hasContolModel = true;
        m_controlModel = std::forward<F>(model);
    }
    inline void clearControlModel() override { m_hasContolModel = false; }
    inline bool hasControlModel() const override { return m_hasContolModel; }

    inline std::function<matrix::Matrix<T> (T timestep, const ControlParams* controlParams)> controlModel() const {
        return m_controlModel;
    }
    
protected:
    inline matrix::Matrix<T> controlModel(T timestep, const ControlParams* controlParams=nullptr) const {
        if(m_hasContolModel){
            return m_controlModel(timestep, controlParams);
        }
        throw NoControlError();
    }

    inline matrix::Vector<T> propagateState_(T timestep, const matrix::Vector<T>& state, const StateTransParams* stateTransParams=nullptr) const {
        return stateTransParams == nullptr ? this->getStateMat(timestep) * state : this->getStateMat(timestep, stateTransParams) * state;
    }

private:
    bool m_hasContolModel = false;
    std::function<matrix::Matrix<T> (T timestep, const ControlParams* controlParams)> m_controlModel;
};
    
} // namespace lager::gncpy::dynamics
