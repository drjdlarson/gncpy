#include <gtest/gtest.h>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/dynamics/DoubleIntegrator.h"
#include "gncpy/measurements/StateObservation.h"
#include "gncpy/filters/Kalman.h"

TEST(FilterTest, SetStateModel) {
    double dt = 0.01;
    lager::gncpy::matrix::Matrix<double> noise({{1.0, 0.0}, {0.0, 1.0}});
    lager::gncpy::dynamics::DoubleIntegrator<double> dynObj(dt);

    lager::gncpy::filters::Kalman<double> filt(dt);

    filt.setStateModel(dynObj, noise);

    SUCCEED();

}

// TEST(FilterTest, SetMeasModel) {
//     double dt = 0.01;
//     lager::gncpy::matrix::Matrix<double> noise({{1.0, 0.0}, {0.0, 1.0}});
//     // std::vector<uint8_t> inds = {0, 1};
//     // lager::gncpy::measurements::StateObservation<double> measObj(std::vector<uint8_t>({0, 1}));
//     // std::unique_ptr<lager::gncpy::measurements::MeasParams> params = std::make_unique<lager::gncpy::measurements::StateObservationParams>(inds);
//     lager::gncpy::measurements::StateObservation<double> measObj;
    


//     lager::gncpy::filters::Kalman<double> filt;

//     filt.setMeasurementModel(measObj, noise);

//     SUCCEED();
// }

// TEST(FilterTest, FilterPredict) {
    
// }

// TEST(FilterTest, FilterCorrect) {
    
// }