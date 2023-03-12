#pragma once
#include <memory>
#include "gncpy/dynamics/Parameters.h"


namespace lager::gncpy::filters{

struct BayesPredictParams {
    std::unique_ptr<lager::gncpy::dynamics::StateTransParams> stateTransParams; 
    std::unique_ptr<lager::gncpy::dynamics::ControlParams> controlParams;
};

} // namespace lager::gncpy::filters