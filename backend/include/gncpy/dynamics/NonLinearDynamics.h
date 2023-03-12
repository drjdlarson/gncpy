#pragma once
#include <functional>
#include "gncpy/dynamics/Parameters.h"
#include "gncpy/dynamics/IDynamics.h"
#include "gncpy/math/Vector.h"


namespace lager::gncpy::dynamics {

template<typename T>
class NonLinearDynamics : public IDynamics<T> {
using control_fun_sig = (T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control, const ControlParams* controlParams);

public:
    matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const StateTransParams* const stateTransParams=nullptr) const override;
    matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control) const override;
    matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control, const StateTransParams* const stateTransParams, const ControlParams* const controlParams, const ConstraintParams* const constraintParams) const final;

    template<typename F>
    inline void setControlModel(F&& model) { 
        m_hasContolModel = true;
        m_controlModel = std::forward<F>(model);
    }
    inline void clearControlModel() override { m_hasContolModel = false; }
    inline bool hasControlModel() const override { return m_hasContolModel; }

    matrix::Matrix<T> getStateMat(T timestep, const StateTransParams* stateTransParams=nullptr) const;
    matrix::Matrix<T> getInputMat(T timestep, const ControlParams* controlParams=nullptr) const

protected:
    inline matrix::Matrix<T> controlModel(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control, const ControlParams* controlParams=nullptr) const {
        if(m_hasContolModel){
            return m_controlModel(timestep, stat, control, controlParams);
        }
        throw NoControlError();
    }

    inline matrix::Vector<T> propagateState_(T timestep, const matrix::Vector<T>& state, const StateTransParams* stateTransParams=nullptr) const {
        //TODO: implement this
    }

private:
    bool m_hasContolModel = false;
    std::function<matrix::Matrix<T> (control_fun_sig)> m_controlModel;
};

} // namespace lager::gncpy::dynamics