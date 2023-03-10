#pragma once
#include "dynamics/ILinearDynamics.h"

namespace lager::gncpy::dynamics {

template<typename T>
class DoubleIntegrator final: ILinearDynamics<T>{
    matrix::Matrix<T> getStateMat(T timestep) const override{
        matrix::Matrix<T> F(4, 4);

        return F;
    }

    matrix::Matrix<T> getInputMat(T timestep) const override {
        
    }
};
    
} // namespace lager::gncpy::dynamics 