#pragma once
#include <memory>
#include <optional>
#include "gncpy/math/Vector.h"
#include "gncpy/filters/Parameters.h"
#include "gncpy/dynamics/IDynamics.h"
#include "gncpy/measurements/IMeasModel.h"

namespace lager::gncpy::filters {

template<typename T>
class IBayesFilter {
public:

    virtual matrix::Vector<T> predict(T timestep, const matrix::Vector<T>& curState, const std::optional<matrix::Vector<T>> controlInput, const BayesPredictParams* params=nullptr) = 0;

    virtual matrix::Vector<T> correct(T timestep, const matrix::Vector<T>& meas, const matrix::Vector<T>& curState, const CorrectParams* params=nullptr) = 0;
    // TODO: fix the remaining interface
    // virtual matrix::Matrix correct(T timestep, const matrix::Matrix& meas, const matrix::Matrix& curState, T& measLikelihood) = 0;

protected:
    std::shared_ptr<dynamics::IDynamics<T>> m_dynObj;
    std::shared_ptr<measurements::IMeasModel<T>> m_measObj;
};


} // namespace lager::gncpy::filters 