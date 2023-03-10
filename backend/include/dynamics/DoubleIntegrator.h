#pragma once
#include "dynamics/ILinearDynamics.h"

namespace lager::gncpy::dynamics {

template<typename T>
class DoubleIntegrator final: public ILinearDynamics<T>{
public:
    inline std::vector<std::string> stateNames() const { return std::vector<std::string>{"x pos", "y pos", "x vel", "y vel"}; };

    explicit DoubleIntegrator(T dt)
    : m_dt(dt) {

    }

    matrix::Matrix<T> getStateMat(T timestep) const override{
        matrix::Matrix<T> F({{1, 0, m_dt, 0},
                             {0, 1, 0, m_dt},
                             {0, 0, 1, 0},
                             {0, 0, 0, 1}});

        return F;
    }

private:
    T m_dt;
};
    
} // namespace lager::gncpy::dynamics 