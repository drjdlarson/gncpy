#pragma once
#include <memory>
#include <optional>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/filters/Parameters.h"
#include "gncpy/dynamics/IDynamics.h"
#include "gncpy/measurements/IMeasModel.h"

namespace lager::gncpy::filters {

template<typename T>
class IBayesFilter {
public:
    virtual matrix::Vector<T> predict(T timestep, const matrix::Vector<T>& curState, const std::optional<matrix::Vector<T>> controlInput, const BayesPredictParams* params=nullptr) = 0;
    virtual matrix::Vector<T> correct(T timestep, const matrix::Vector<T>& meas, const matrix::Vector<T>& curState, T& measFitProb, const BayesCorrectParams* params=nullptr) = 0;
    virtual void setStateModel(std::shared_ptr<dynamics::IDynamics<T>> dynObj, matrix::Matrix<T> procNoise) = 0;
    virtual void setMeasurementModel(std::shared_ptr<measurements::IMeasModel<T>> measObj, matrix::Matrix<T> measNoise) = 0;

    virtual std::shared_ptr<dynamics::IDynamics<T>> dynamicsModel() const = 0;
    virtual std::shared_ptr<measurements::IMeasModel<T>> measurementModel() const = 0;

    matrix::Matrix<T> cov;

};


} // namespace lager::gncpy::filters 