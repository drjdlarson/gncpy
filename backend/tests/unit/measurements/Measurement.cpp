#include <math.h>
#include <gtest/gtest.h>
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Vector.h"
#include "gncpy/math/Math.h"
#include "gncpy/measurements/RangeAndBearing.h"
#include "gncpy/measurements/Exceptions.h"


TEST(MeasurementTest, RangeBearingMeasure) {
    lager::gncpy::matrix::Vector x({3.0, 4.0, 1.0});
    lager::gncpy::measurements::RangeAndBearing<double> sensor;
    lager::gncpy::matrix::Vector exp({5.0, atan2(4.0, 3.0)});

    EXPECT_THROW(sensor.getMeasMat(x), lager::gncpy::measurements::BadParams);

    std::unique_ptr<lager::gncpy::measurements::MeasParams> params = std::make_unique<lager::gncpy::measurements::RangeBearingParams>(0, 1);

    lager::gncpy::matrix::Vector out = sensor.measure(x, params);

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

    EXPECT_THROW(sensor.getMeasMat(x), lager::gncpy::measurements::BadParams);

    std::unique_ptr<lager::gncpy::measurements::MeasParams> params = std::make_unique<lager::gncpy::measurements::RangeBearingParams>(0, 1);

    lager::gncpy::matrix::Matrix out = sensor.getMeasMat(x, params);

    auto [nRows, nCols] = out.shape();

    EXPECT_EQ(exp.numRows(), nRows);
    EXPECT_EQ(exp.numCols(), nCols);

    for (uint8_t ii=0;ii<nRows;ii++) {
        for (uint8_t jj=0;jj<nCols;jj++) {
            EXPECT_NEAR(exp(ii, jj), out(ii, jj), 1e-6);
        }
    }
    SUCCEED();
}