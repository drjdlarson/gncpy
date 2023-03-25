#pragma once
#include <memory>
#include <optional>
#include "gncpy/filters/IBayesFilter.h"
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Math.h"
#include "gncpy/filters/Parameters.h"
#include "gncpy/dynamics/ILinearDynamics.h"
#include "gncpy/measurements/ILinearMeasModel.h"
#include "gncpy/Exceptions.h"
#include "gncpy/Utilities.h"

namespace lager::gncpy::filters {

template<typename T>
class Kalman : public IBayesFilter<T> {
public:
    matrix::Vector<T> predict(T timestep, const matrix::Vector<T>& curState, [[maybe_unused]] const std::optional<matrix::Vector<T>> controlInput, const BayesPredictParams* params=nullptr) override {
        if (params != nullptr && !utilities::instanceof<BayesPredictParams>(params)) {
            throw exceptions::BadParams("Params must be BayesPredictParams");
        }
        matrix::Matrix<T> stateMat = std::dynamic_pointer_cast<dynamics::ILinearDynamics<T>>(this->dynamicsModel())->getStateMat(timestep, params->stateTransParams.get());
        this->cov = stateMat * this->cov * stateMat.transpose() + this->m_procNoise;

        return this->m_dynObj->propagateState(timestep, curState, params->stateTransParams.get());
    }

    matrix::Vector<T> correct(T timestep, const matrix::Vector<T>& meas, const matrix::Vector<T>& curState, T& measFitProb, const BayesCorrectParams* params=nullptr) override {
        if (params != nullptr && !utilities::instanceof<BayesCorrectParams>(params)) {
            throw exceptions::BadParams("Params must be BayesCorrectParams");
        }

        matrix::Vector<T> estMeas = this->measurementModel()->measure(curState, params->measParams.get());
        matrix::Matrix<T> measMat = this->measurementModel()->getMeasMat(curState, params->measParams.get());

        matrix::Matrix<T> inovCov = measMat * this->cov * measMat.transpose();

        matrix::Matrix<T> kalmanGain = this->cov * measMat.transpose() * inovCov.inverse();

        matrix::Vector<T> inov = meas - estMeas;
        this->cov -= kalmanGain * measMat * this->cov;

        measFitProb = math::calcGaussianPDF(meas, estMeas, inovCov);

        return curState + kalmanGain * inov;
    }

    void setStateModel(std::shared_ptr<dynamics::IDynamics<T>> dynObj, matrix::Matrix<T> procNoise) override {
        if (!dynObj || !utilities::instanceof<dynamics::ILinearDynamics<T>>(dynObj.get())) {
            throw exceptions::TypeError("dynObj must be a derived class of ILinearDynamics");
        }
        if(procNoise.numRows() != procNoise.numCols()) {
            throw exceptions::BadParams("Process noise must be square");
        }
        if(procNoise.numRows() != dynObj->stateNames().size()){
            throw exceptions::BadParams("Process nosie size does not match they dynamics model dimension");
        }

        this->m_dynObj = std::dynamic_pointer_cast<dynamics::ILinearDynamics<T>>(dynObj);
        this->m_procNoise = procNoise;
 
    }

    void setMeasurementModel(std::shared_ptr<measurements::IMeasModel<T>> measObj, matrix::Matrix<T> measNoise) override{
        if (!measObj || !utilities::instanceof<measurements::ILinearMeasModel<T>>(measObj.get())) {
            throw exceptions::TypeError("measObj must be a derived class of ILinearMeasModel");
        }

        if(measNoise.numRows() != measNoise.numCols()) {
            throw exceptions::BadParams("Measurement noise must be squqre");
        }
        
        this->m_measObj = std::dynamic_pointer_cast<measurements::ILinearMeasModel<T>>(measObj);
        this->m_measNoise = measNoise;
    }

    inline std::shared_ptr<dynamics::IDynamics<T>> dynamicsModel() const override {
        if(m_dynObj) {
            return m_dynObj;
        } else {
            throw exceptions::TypeError("Dynamics model is unset");
        }
    }

    inline std::shared_ptr<measurements::IMeasModel<T>> measurementModel() const override {
        if(m_measObj) {
            return m_measObj;
        } else {
            throw exceptions::TypeError("Measurement model is unset");
        }
    }


private:
    matrix::Matrix<T> m_measNoise;
    matrix::Matrix<T> m_procNoise;

    std::shared_ptr<dynamics::ILinearDynamics<T>> m_dynObj;
    std::shared_ptr<measurements::ILinearMeasModel<T>> m_measObj;

};
    
} // namespace lager::gncpy::filters
