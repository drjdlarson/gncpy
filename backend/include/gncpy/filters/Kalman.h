#pragma once
#include <optional>
#include "gncpy/filters/IBayesFilter.h"
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/filters/Parameters.h"
#include "gncpy/dynamics/ILinearDynamics.h"
#include "gncpy/measurements/ILinearMeasModel.h"

namespace lager::gncpy::filters {

template<typename T>
class Kalman : public IBayesFilter<T> {
public:

    Kalman(T dt) 
    : M_dt(dt), 
      m_cov(2, 2),
      M_measNoise(2, 2),
      M_procNoise(2, 2) {

    }

    auto setStateModel(dynamics::ILinearDynamics<T>& dynObj, matrix::Matrix<T> procNoise) {
        *this->M_dynObj = dynObj;
        this->M_procNoise = procNoise;

    }

    auto setMeasurementModel(measurements::ILinearMeasModel<T>& measObj, matrix::Matrix<T> measNoise) {
        *this->M_measObj = measObj;
        this->M_measNoise = measNoise;
    }

    matrix::Vector<T> predict(T timestep, const matrix::Vector<T>& curState, const std::optional<matrix::Vector<T>> controlInput, const BayesPredictParams* params=nullptr) override {
        this->M_dynObj->propagateState(timestep, curState, params->stateTransParams.get());
        matrix::Matrix stateMat = this->M_dynObj->getStateMat(timestep, params->stateTransParams.get());

        this->m_cov = stateMat * this->m_cov * stateMat.transpose() + this->M_procNoise;

        return matrix::Vector<T>(curState);
    }

    matrix::Vector<T> correct([[maybe_unused]] T timestep, [[maybe_unused]] const matrix::Vector<T>& meas, [[maybe_unused]] const matrix::Vector<T>& curState, [[maybe_unused]] const CorrectParams* params=nullptr) override {
    // matrix::Vector<T> correct(T timestep, const matrix::Vector<T>& meas, const matrix::Vector<T>& curState, const CorrectParams* params=nullptr) override {
    //     matrix::Vector est_meas = this->M_measObj->measure(curState, params->measParams.get());
    //     matrix::Matrix measMat = this->M_measObj->getMeasMat(curState, params->measParams.get());

    //     matrix::Matrix inovCov = measMat * this->m_cov * measMat.transpose();

    //     matrix::Matrix kalmanGain = this->m_cov * measMat.transpose() * inovCov.inverse();

    //     curState = curState + kalmanGain * (meas - est_meas);
    //     this->m_cov = this->m_cov - kalmanGain * measMat * this->m_cov;

        
    //     return matrix::Vector<T>(curState);
    }

    private:

        T M_dt;
        matrix::Matrix<T> m_cov;
        matrix::Matrix<T> M_measNoise;
        matrix::Matrix<T> M_procNoise;
        std::shared_ptr<dynamics::ILinearDynamics<T>> M_dynObj;
        std::shared_ptr<measurements::ILinearMeasModel<T>> M_measObj;

};
    
} // namespace lager::gncpy::filters
