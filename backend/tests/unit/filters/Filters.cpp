#include <gtest/gtest.h>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/dynamics/DoubleIntegrator.h"
#include "gncpy/measurements/StateObservation.h"
#include "gncpy/filters/Kalman.h"
#include "gncpy/measurements/Parameters.h"
#include "gncpy/dynamics/Parameters.h"

TEST(FilterTest, SetStateModel) {
    double dt = 0.01;
    lager::gncpy::matrix::Matrix<double> noise({{1.0, 0.0}, {0.0, 1.0}});
    auto dynObj = std::make_shared<lager::gncpy::dynamics::DoubleIntegrator<double>>(dt);

    lager::gncpy::filters::Kalman<double> filt(dt);

    filt.setStateModel(dynObj, noise);

    SUCCEED();

}

TEST(FilterTest, SetMeasModel) {
    double dt = 0.01;
    lager::gncpy::matrix::Matrix<double> noise({{1.0, 0.0}, {0.0, 1.0}});
    std::vector<uint8_t> inds = {0, 1};
    // lager::gncpy::measurements::StateObservation<double> measObj(std::vector<uint8_t>({0, 1}));
    // std::unique_ptr<lager::gncpy::measurements::MeasParams> params = std::make_unique<lager::gncpy::measurements::StateObservationParams>(inds);
    
    auto measObj = std::make_shared<lager::gncpy::measurements::StateObservation<double>>();
    


    lager::gncpy::filters::Kalman<double> filt(dt);

    filt.setMeasurementModel(measObj, noise);

    SUCCEED();
}

TEST(FilterTest, GetSetCovariance) {
    double dt = 0.01;
    lager::gncpy::matrix::Matrix covariance({{0.1, 0.0, 0.0, 0.0}, {0.0, 0.1, 0.0, 0.0}, {0.0, 0.0, 0.01, 0.0},{0.0, 0.0, 0.0, 0.01}});
    lager::gncpy::filters::Kalman<double> filt(dt);

    filt.setCovariance(covariance);
    lager::gncpy::matrix::Matrix newCov = filt.getCovariance();

    for (uint8_t ii=0;ii<4;ii++) {
        for (uint8_t jj=0;jj<4;jj++) {
            EXPECT_EQ(newCov(ii, jj), covariance(ii, jj));
        }
    }

    SUCCEED();

}

TEST(FilterTest, FilterPredict) {
    double dt = 1.0;
    lager::gncpy::matrix::Matrix<double> noise({{0.1, 0.0, 0.0, 0.0}, {0.0, 0.1, 0.0, 0.0}, {0.0, 0.0, 0.01, 0.0},{0.0, 0.0, 0.0, 0.01}});

    auto dynObj = std::make_shared<lager::gncpy::dynamics::DoubleIntegrator<double>>(dt);
    auto measObj = std::make_shared<lager::gncpy::measurements::StateObservation<double>>();
    lager::gncpy::matrix::Vector state({1.0, 2.0, 1.0, 1.0});

    lager::gncpy::matrix::Vector control({0.0, 0.0});
    // lager::gncpy::matrix::Vector control(2);
    lager::gncpy::matrix::Vector exp({2.0, 3.0, 1.0, 1.0});

    std::vector<uint8_t> inds = {0, 1};
    auto params = lager::gncpy::measurements::StateObservationParams(inds);
    lager::gncpy::filters::Kalman<double> filt(dt);

    filt.setMeasurementModel(measObj, noise);
    filt.setStateModel(dynObj, noise);

    auto out = filt.predict(0.0, state, control, nullptr);

    for(uint8_t ii=0;ii<exp.size();ii++) {
        EXPECT_EQ(exp(ii), out(ii));
    }

    SUCCEED();

}

// TEST(FilterTest, FilterCorrect) {
    
// }