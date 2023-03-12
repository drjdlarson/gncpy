#include <iostream>
#include <memory>
#include <optional>
#include "gncpy/math/Vector.h"
#include "gncpy/filters/Kalman.h"
#include "gncpy/filters/Parameters.h"
#include "gncpy/dynamics/Parameters.h"


int main() {
    auto fpParams = std::make_unique<lager::gncpy::filters::BayesPredictParams>();
    std::optional<lager::gncpy::matrix::Vector<float>> u;

    // NOTE: can do either of the following, but if move is used then sParams can not be used later!!
    auto sParams = std::make_unique<lager::gncpy::dynamics::StateTransParams>();
    fpParams->stateTransParams = std::move(sParams);
    // fpParams->stateTransParams = std::make_unique<lager::gncpy::dynamics::StateTransParams>();

    lager::gncpy::matrix::Vector<float> x({1, 2, 3});
    lager::gncpy::filters::Kalman<float> filt;

    filt.predict(0.1, x, u, fpParams);
    filt.predict(0.1, x, u);
    
    return 0;
}