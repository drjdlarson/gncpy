#pragma once
#include <optional>
#include "gncpy/filters/IBayesFilter.h"
#include "gncpy/math/Vector.h"

namespace lager::gncpy::filters {

template<typename T>
class Kalman : public IBayesFilter<T> {
public:
    matrix::Vector<T> predict(T timestep, const matrix::Vector<T>& curState, const std::optional<matrix::Vector<T>> controlInput, const std::unique_ptr<BayesPredictParams>& params=std::make_unique<BayesPredictParams>()) override {
        return matrix::Vector<T>(curState);
    }

};
    
} // namespace lager::gncpy::filters
