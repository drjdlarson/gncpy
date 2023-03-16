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
      m_measNoise(2, 2),
      m_procNoise(2, 2) {

    }

    void setStateModel(dynamics::ILinearDynamics<T>& dynObj, matrix::Matrix<T> procNoise) {
        *this->m_dynObj = dynObj;
        this->m_procNoise = procNoise;

    }

    void setMeasurementModel(measurements::ILinearMeasModel<T>& measObj, matrix::Matrix<T> measNoise) {
        *this->m_measObj = measObj;
        this->m_measNoise = measNoise;
    }

    matrix::Vector<T> predict(T timestep, const matrix::Vector<T>& curState, [[maybe_unused]] const std::optional<matrix::Vector<T>> controlInput, const BayesPredictParams* params=nullptr) override {
        this->m_dynObj->propagateState(timestep, curState, params->stateTransParams.get());
        matrix::Matrix<T> stateMat = this->m_dynObj->getStateMat(timestep, params->stateTransParams.get());

        this->m_cov = stateMat * this->m_cov * stateMat.transpose() + this->m_procNoise;

        return matrix::Vector<T>(curState);
    }

    matrix::Vector<T> correct([[maybe_unused]] T timestep, [[maybe_unused]] const matrix::Vector<T>& meas, [[maybe_unused]] const matrix::Vector<T>& curState, [[maybe_unused]] const CorrectParams* params=nullptr) override {
    // matrix::Vector<T> correct(T timestep, const matrix::Vector<T>& meas, const matrix::Vector<T>& curState, const CorrectParams* params=nullptr) override {
    //     matrix::Vector est_meas = this->m_measObj->measure(curState, params->measParams.get());
    //     matrix::Matrix measMat = this->m_measObj->getMeasMat(curState, params->measParams.get());

    //     matrix::Matrix inovCov = measMat * this->m_cov * measMat.transpose();

    //     matrix::Matrix kalmanGain = this->m_cov * measMat.transpose() * inovCov.inverse();

    //     curState = curState + kalmanGain * (meas - est_meas);
    //     this->m_cov = this->m_cov - kalmanGain * measMat * this->m_cov;

        
        return matrix::Vector<T>(curState);
    }

    private:
        T M_dt;
        matrix::Matrix<T> m_cov;
        matrix::Matrix<T> m_measNoise;
        matrix::Matrix<T> m_procNoise;
        std::shared_ptr<dynamics::ILinearDynamics<T>> m_dynObj;
        std::shared_ptr<measurements::ILinearMeasModel<T>> m_measObj;

};
    
} // namespace lager::gncpy::filters
