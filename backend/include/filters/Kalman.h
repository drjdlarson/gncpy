#pragma once
#include "filters/IBayesFilter.h"

namespace lager::gncpy::filters {

template<typename T>
class Kalman : public IBayesFilter<T> {
    Kalman();

};
    
} // namespace lager::gncpy::filters
