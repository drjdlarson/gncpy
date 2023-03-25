#pragma once
#include <functional>
#include "gncpy/dynamics/Exceptions.h"
#include "gncpy/dynamics/Parameters.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Vector.h"

namespace lager::gncpy::dynamics {

template<typename T>
class IDynamics {
public:
    virtual matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const StateTransParams* const stateTransParams=nullptr) const = 0;
    virtual matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control) const = 0;
    virtual matrix::Vector<T> propagateState(T timestep, const matrix::Vector<T>& state, const matrix::Vector<T>& control, const StateTransParams* const stateTransParams, const ControlParams* const controlParams, const ConstraintParams* const constraintParams) const = 0;

    virtual void clearControlModel() = 0;
    virtual bool hasControlModel() const = 0;
    
    virtual std::vector<std::string> stateNames() const = 0;

    template<typename F>
    inline void setStateConstraints(F&& constrants) {
        m_hasStateConstraint = false;
        m_stateConstraints = std::forward<F>(constrants);
    }
    inline void clearStateConstraints() { m_hasStateConstraint = false; }
    inline bool hasStateConstraint() const { return m_hasStateConstraint; }

    inline std::function<void (T timestep, matrix::Vector<T>& state, const ConstraintParams* const constraintParams)> stateConstraints() const {
        return m_stateConstraints;
    }

protected:
    inline void stateConstraint(T timestep, matrix::Vector<T>& state, const ConstraintParams* const constraintParams=nullptr) const {
        if(m_hasStateConstraint) {
            m_stateConstraints(timestep, state, constraintParams);
        }
        throw NoStateConstraintError();
    }

private:
    bool m_hasStateConstraint = false;
    std::function<void (T timestep, matrix::Vector<T>& state, const ConstraintParams* const constraintParams)> m_stateConstraints;
};
    
} // namespace lager::gncpy::dynamics 