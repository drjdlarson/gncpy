#pragma once
#include "gncpy/dynamics/ILinearDynamics.h"

namespace lager::gncpy::dynamics {

template<typename T>
class DoubleIntegrator final: public ILinearDynamics<T>{
public:
    inline std::vector<std::string> stateNames() const { return std::vector<std::string>{"x pos", "y pos", "x vel", "y vel"}; };

    explicit DoubleIntegrator(T dt)
    : m_dt(dt) {

    }

    matrix::Matrix<T> getStateMat([[maybe_unused]] T timestep, [[maybe_unused]] const StateTransParams* const stateTransParams=nullptr) const override{
        matrix::Matrix<T> F({{1, 0, m_dt, 0},
                             {0, 1, 0, m_dt},
                             {0, 0, 1, 0},
                             {0, 0, 0, 1}});

        return F;
    }

    inline T dt() const { return m_dt; }
    inline void setDt(T dt) { m_dt = dt; }

private:
    T m_dt;
};
    
} // namespace lager::gncpy::dynamics 