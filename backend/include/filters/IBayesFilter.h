#pragma once
#include "Matrix.h"

namespace lager::gncpy::filters {

template<typename T>
class IBayesFilter {
public:
    virtual gncpy::matrix::Matrix predict(T timestep, const gncpy::matrix::Matrix& curState) = 0;
    virtual gncpy::matrix::Matrix correct(T timestep, const gncpy::matrix::Matrix& meas, const gncpy::matrix::Matrix& curState, T& measLikelihood) = 0;

};


} // namespace lager::gncpy::filters 