#pragma once
#include <memory>
#include "gncpy/dynamics/Parameters.h"
#include "gncpy/measurements/Parameters.h"


namespace lager::gncpy::filters{

struct BayesPredictParams {
    std::unique_ptr<lager::gncpy::dynamics::StateTransParams> stateTransParams; 
    std::unique_ptr<lager::gncpy::dynamics::ControlParams> controlParams;
};

struct CorrectParams {
    std::unique_ptr<lager::gncpy::measurements::MeasParams> measParams;
};

} // namespace lager::gncpy::filters