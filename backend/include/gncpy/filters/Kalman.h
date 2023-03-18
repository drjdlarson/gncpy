#pragma once
#include <optional>
#include "gncpy/filters/IBayesFilter.h"
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/filters/Parameters.h"
#include "gncpy/dynamics/ILinearDynamics.h"
#include "gncpy/measurements/ILinearMeasModel.h"
#include "gncpy/Exceptions.h"
#include "gncpy/Utilities.h"

namespace lager::gncpy::filters {

template<typename T>
class Kalman : public IBayesFilter<T> {
public:

    Kalman(T dt) 
    : M_dt(dt), 
      m_cov(),
      m_measNoise(),
      m_procNoise() {

    }

    void setStateModel(std::shared_ptr<dynamics::ILinearDynamics<T>> dynObj, matrix::Matrix<T> procNoise) {
        if (!dynObj || !utilities::instanceof<dynamics::ILinearDynamics<T>>(dynObj.get())) {
            throw exceptions::TypeError("dynObj must be a derived class of ILinearDynamics");
        }
        this->m_dynObj = dynObj;
        this->m_procNoise = procNoise;

    }

    void setMeasurementModel(std::shared_ptr<measurements::ILinearMeasModel<T>> measObj, matrix::Matrix<T> measNoise) {
        if (!measObj || !utilities::instanceof<measurements::ILinearMeasModel<T>>(measObj.get())) {
            throw exceptions::TypeError("measObj must be a derived class of ILinearMeasModel");
        }
        
        this->m_measObj = measObj;
        this->m_measNoise = measNoise;
    }

    void setCovariance(matrix::Matrix<T> newCov) {
        this->m_cov = newCov;
    }

    matrix::Matrix<T>  getCovariance() {
        return this->m_cov;
    }
    matrix::Vector<T> predict(T timestep, const matrix::Vector<T>& curState, [[maybe_unused]] const std::optional<matrix::Vector<T>> controlInput, const BayesPredictParams* params=nullptr) override {
        if (params != nullptr && !utilities::instanceof<BayesPredictParams>(params)) {
            throw exceptions::BadParams("Params must be BayesPredictParams");
        }
        if (this->m_dynObj == nullptr) {
            throw exceptions::TypeError("dynamic object not initialized.");
        }
        matrix::Vector<T> newState(curState.size());
        matrix::Matrix<T> stateMat(curState.size(), curState.size());
        if (params == nullptr) {
            newState = this->m_dynObj->propagateState(timestep, curState);
            stateMat = dynamic_cast<dynamics::ILinearDynamics<T>*>(this->m_dynObj.get())->getStateMat(timestep);
        }
        else {
            newState = this->m_dynObj->propagateState(timestep, curState, params->stateTransParams.get());
            stateMat = dynamic_cast<dynamics::ILinearDynamics<T>*>(this->m_dynObj.get())->getStateMat(timestep, params->stateTransParams.get());
        }
        this->m_cov = stateMat * this->m_cov * stateMat.transpose() + this->m_procNoise;

        return matrix::Vector<T>(newState);
    }

    // matrix::Vector<T> correct([[maybe_unused]] T timestep, [[maybe_unused]] const matrix::Vector<T>& meas, [[maybe_unused]] const matrix::Vector<T>& curState, [[maybe_unused]] const CorrectParams* params=nullptr) override {
    matrix::Vector<T> correct(T timestep, const matrix::Vector<T>& meas, const matrix::Vector<T>& curState, const CorrectParams* params=nullptr) override {
        matrix::Vector<T> est_meas = this->m_measObj->measure(curState, params->measParams.get());
        matrix::Matrix<T> measMat = this->m_measObj->getMeasMat(curState, params->measParams.get());

        matrix::Matrix<T> inovCov = measMat * this->m_cov * measMat.transpose();

        matrix::Matrix<T> kalmanGain = this->m_cov * measMat.transpose() * inovCov.inverse();

        matrix::Vector<T> inov = meas - est_meas;
        matrix::Vector<T> newState(curState.size());
        newState = curState + kalmanGain * inov;
        this->m_cov = this->m_cov - kalmanGain * measMat * this->m_cov;

        
        return newState;
    }

    private:
        T M_dt;
        matrix::Matrix<T> m_cov;
        matrix::Matrix<T> m_measNoise;
        matrix::Matrix<T> m_procNoise;

};
    
} // namespace lager::gncpy::filters
