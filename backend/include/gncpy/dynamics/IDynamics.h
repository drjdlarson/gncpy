#pragma once
#include <functional>
#include "gncpy/dynamics/Exceptions.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Vector.h"

namespace lager::gncpy::dynamics {

template<typename T>
class IDynamics {
public:
    virtual matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control) const = 0;
    virtual matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state) const = 0;

    virtual matrix::Matrix<T> getStateMat(T timestep) const = 0;
    virtual matrix::Matrix<T> getInputMat(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control) const = 0;

    template<typename F>
    inline void setControlModel(F&& model) { 
        m_hasContolModel = true;
        m_controlModel = std::forward<F>(model);
    }
    inline void clearControlModel() { m_hasContolModel = false; }
    inline matrix::Matrix<T> controlModel(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control) const {
        if(m_hasContolModel){
            return m_controlModel(timestep, state, control);
        }
        throw NoControlError();
    }


    template<typename F>
    inline void setStateConstraints(F&& constrants) {
        m_hasStateConstraint = false;
        m_stateConstraints = std::forward<F>(constrants);
    }
    inline void clearStateConstraints() { m_hasStateConstraint = false; }
    inline void stateConstraint(T timestep, matrix::Vector<T>& state) const {
        if(m_hasStateConstraint) {
            m_stateConstraints(timestep, state);
        }
        throw NoStateConstraintError();
    }

    virtual std::vector<std::string> stateNames() const = 0;

    inline bool hasControlModel() const { return m_hasContolModel; }
    inline bool hasStateConstraint() const { return m_hasStateConstraint; }

private:
    bool m_hasContolModel = false;
    bool m_hasStateConstraint = false;

    std::function<matrix::Matrix<T> (T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control)> m_controlModel;
    std::function<void (T timestep, matrix::Vector<T>& state)> m_stateConstraints;
};
    
} // namespace lager::gncpy::dynamics 