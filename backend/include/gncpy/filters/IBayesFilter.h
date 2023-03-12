#pragma once
#include <memory>
#include <optional>
#include "gncpy/math/Vector.h"
#include "gncpy/filters/Parameters.h"


namespace lager::gncpy::filters {

template<typename T>
class IBayesFilter {
public:
    virtual matrix::Vector<T> predict(T timestep, const matrix::Vector<T>& curState, const std::optional<matrix::Vector<T>> controlInput, const std::unique_ptr<BayesPredictParams>& params=std::make_unique<BayesPredictParams>()) = 0;


    // TODO: fix the remaining interface
    // virtual matrix::Matrix correct(T timestep, const matrix::Matrix& meas, const matrix::Matrix& curState, T& measLikelihood) = 0;

};


} // namespace lager::gncpy::filters 