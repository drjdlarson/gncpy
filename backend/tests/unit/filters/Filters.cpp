#include <gtest/gtest.h>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/dynamics/DoubleIntegrator.h"
#include "gncpy/measurements/StateObservation.h"
#include "gncpy/filters/Kalman.h"
#include "gncpy/measurements/Parameters.h"
#include "gncpy/dynamics/Parameters.h"
#include "gncpy/filters/Parameters.h"

TEST(FilterTest, SetStateModel) {
    double dt = 0.01;
    lager::gncpy::matrix::Matrix<double> noise({{1.0, 0.0, 0, 0}, {0.0, 1.0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
    auto dynObj = std::make_shared<lager::gncpy::dynamics::DoubleIntegrator<double>>(dt);

    lager::gncpy::filters::Kalman<double> filt;

    filt.setStateModel(dynObj, noise);

    SUCCEED();

}

TEST(FilterTest, SetMeasModel) {
    lager::gncpy::matrix::Matrix<double> noise({{1.0, 0.0}, {0.0, 1.0}});
    auto measObj = std::make_shared<lager::gncpy::measurements::StateObservation<double>>();

    lager::gncpy::filters::Kalman<double> filt;

    filt.setMeasurementModel(measObj, noise);

    SUCCEED();
}

TEST(FilterTest, GetSetCovariance) {
    lager::gncpy::matrix::Matrix covariance({{0.1, 0.0, 0.0, 0.0}, {0.0, 0.1, 0.0, 0.0}, {0.0, 0.0, 0.01, 0.0},{0.0, 0.0, 0.0, 0.01}});
    lager::gncpy::filters::Kalman<double> filt;

    filt.setCovariance(covariance);
    lager::gncpy::matrix::Matrix newCov = filt.covariance();

    for (uint8_t ii = 0; ii < 4; ii++) {
        for (uint8_t jj = 0; jj < 4; jj++) {
            EXPECT_EQ(newCov(ii, jj), covariance(ii, jj));
        }
    }

    SUCCEED();

}

TEST(FilterTest, FilterPredict) {
    double dt = 1.0;
    lager::gncpy::matrix::Matrix<double> noise({{0.1, 0.0, 0.0, 0.0}, {0.0, 0.1, 0.0, 0.0}, {0.0, 0.0, 0.01, 0.0},{0.0, 0.0, 0.0, 0.01}});

    auto dynObj = std::make_shared<lager::gncpy::dynamics::DoubleIntegrator<double>>(dt);
    lager::gncpy::matrix::Vector state({1.0, 2.0, 1.0, 1.0});

    lager::gncpy::matrix::Vector control({0.0, 0.0});
    lager::gncpy::matrix::Vector exp({2.0, 3.0, 1.0, 1.0});

    auto predParams = lager::gncpy::filters::BayesPredictParams();
    lager::gncpy::filters::Kalman<double> filt;
    lager::gncpy::matrix::Matrix cov({{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}});
    filt.setCovariance(cov);

    filt.setStateModel(dynObj, noise);

    auto out = filt.predict(0.0, state, control, &predParams);

    for(uint8_t ii = 0; ii < exp.size(); ii++) {
        EXPECT_EQ(exp(ii), out(ii));
    }

    SUCCEED();

}

TEST(FilterTest, FilterCorrect) {
    lager::gncpy::matrix::Matrix<double> noise({{0.1, 0.0, 0.0, 0.0}, {0.0, 0.1, 0.0, 0.0}, {0.0, 0.0, 0.01, 0.0},{0.0, 0.0, 0.0, 0.01}});

    auto measObj = std::make_shared<lager::gncpy::measurements::StateObservation<double>>();
    lager::gncpy::matrix::Vector state({1.0, 2.0, 1.0, 1.0});

    lager::gncpy::matrix::Vector exp({2.0, 3.0, 1.0, 1.0});

    std::vector<uint8_t> inds = {0, 1};
    auto corrParams = lager::gncpy::filters::BayesCorrectParams();
    corrParams.measParams = std::make_shared<lager::gncpy::measurements::StateObservationParams>(inds);
    lager::gncpy::filters::Kalman<double> filt;

    lager::gncpy::matrix::Matrix cov({{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}});
    filt.setCovariance(cov);
    
    filt.setMeasurementModel(measObj, noise);

    auto meas = measObj->measure(exp, corrParams.measParams.get());

    double measFitProb;
    auto out = filt.correct(0.0, meas, exp, measFitProb, &corrParams);

    for(uint8_t ii=0;ii<exp.size();ii++) {
        EXPECT_EQ(exp(ii), out(ii));
    }

    SUCCEED();
}