#include <math.h>
#include <gtest/gtest.h>
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Vector.h"
#include "gncpy/math/Math.h"
#include "gncpy/measurements/RangeAndBearing.h"
#include "gncpy/measurements/StateObservation.h"
#include "gncpy/Exceptions.h"

TEST(MeasurementTest, StateObservationMeasure) {
    lager::gncpy::matrix::Vector x({3.0, 4.0, 1.0});
    lager::gncpy::measurements::StateObservation<double> sensor;
    std::vector<uint8_t> inds = {0, 1, 2};

    EXPECT_THROW(sensor.measure(x, nullptr), lager::gncpy::exceptions::BadParams);

    auto params = lager::gncpy::measurements::StateObservationParams(inds);

    lager::gncpy::matrix::Vector out = sensor.measure(x, &params);


    EXPECT_EQ(3, out.size());

    for (uint8_t ii=0; ii < out.size(); ii++) {
        EXPECT_DOUBLE_EQ(x(ii), out(ii));
    }

    SUCCEED();
}

TEST(MeasurementTest, StateObservationMeasMat) {
    lager::gncpy::matrix::Vector x({3.0, 4.0, 1.0});
    lager::gncpy::measurements::StateObservation<double> sensor;
    std::vector<uint8_t> inds = {0, 1, 2};
    auto params = lager::gncpy::measurements::StateObservationParams(inds);

    EXPECT_THROW(sensor.getMeasMat(x, nullptr), lager::gncpy::exceptions::BadParams);

    lager::gncpy::matrix::Matrix out = sensor.getMeasMat(x, &params);
    lager::gncpy::matrix::Matrix exp({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});

    EXPECT_EQ(9, out.size());

    for (uint8_t r=0;r<x.size();r++) {
        for (uint8_t c=0;c<inds.size();c++) {
            EXPECT_EQ(exp(r, c), out(r, c));
        }
    }

    SUCCEED();
}

TEST(MeasurementTest, RangeBearingMeasure) {
    lager::gncpy::matrix::Vector x({3.0, 4.0, 1.0});
    lager::gncpy::measurements::RangeAndBearing<double> sensor;
    lager::gncpy::matrix::Vector exp({5.0, atan2(4.0, 3.0)});

    EXPECT_THROW(sensor.measure(x, nullptr), lager::gncpy::exceptions::BadParams);

    lager::gncpy::measurements::RangeAndBearingParams params(0, 1);

    lager::gncpy::matrix::Vector out = sensor.measure(x, &params);

    EXPECT_EQ(2, out.size());

    for (uint8_t ii=0; ii < out.size(); ii++) {
        EXPECT_DOUBLE_EQ(exp(ii), out(ii));
    }

    SUCCEED();
}

TEST(MeasurementTest, RangeBearingMeasMat) {
    lager::gncpy::matrix::Vector x({3.0, 4.0, 1.0});
    lager::gncpy::measurements::RangeAndBearing<double> sensor;
    lager::gncpy::matrix::Matrix exp({{0.6, 0.8, 0.0},{-0.16, 0.12, 0.0}});

    EXPECT_THROW(sensor.getMeasMat(x), lager::gncpy::exceptions::BadParams);

    lager::gncpy::measurements::RangeAndBearingParams params(0, 1);

    lager::gncpy::matrix::Matrix res = sensor.getMeasMat(x, &params);

    uint8_t nRows = res.shape()[0];
    uint8_t nCols = res.shape()[1];

    EXPECT_EQ(exp.numRows(), nRows);
    EXPECT_EQ(exp.numCols(), nCols);

    for (uint8_t ii=0;ii<nRows;ii++) {
        for (uint8_t jj=0;jj<nCols;jj++) {
            EXPECT_NEAR(exp(ii, jj), res(ii, jj), 1e-6);
        }
    }
    SUCCEED();
}