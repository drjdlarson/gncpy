#pragma once
#include <math.h>
#include <functional>
#include <vector>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/measurements/INonLinearMeasModel.h"
#include "gncpy/measurements/Parameters.h"
#include "gncpy/Exceptions.h"
#include "gncpy/Utilities.h"


namespace lager::gncpy::measurements{

class RangeAndBearingParams final: public MeasParams {
public: 
    RangeAndBearingParams(uint8_t xInd, uint8_t yInd) 
    : xInd(xInd), 
    yInd(yInd) {
        
    }

    uint8_t xInd;
    uint8_t yInd;
};

template<typename T>
class RangeAndBearing final: public INonLinearMeasModel<T> {   
protected:
    std::vector<std::function<T (const matrix::Vector<T>&)>> getMeasFuncLst(const MeasParams* params) const override {
        auto h1 = [this, params](const matrix::Vector<T>& x) { return this->range(x, params); };
        auto h2 = [this, params](const matrix::Vector<T>& x) { return this->bearing(x, params); };
        return std::vector<std::function<T (const matrix::Vector<T>&)>>({h1, h2});
    }
private:
    static T range (const matrix::Vector<T>& state, const MeasParams* params=nullptr) {
        if (!params) {
            throw exceptions::BadParams("Range and Bearing requires parameters.");
        }
        if (!utilities::instanceof<RangeAndBearingParams>(params)) {
            throw exceptions::BadParams("params type must be RangeAndBearingParams.");
        }
        auto ptr = dynamic_cast<const RangeAndBearingParams*>(params);

        return sqrt(state(ptr->xInd) * state(ptr->xInd) + state(ptr->yInd) * state(ptr->yInd));
    }
    static T bearing (const matrix::Vector<T>& state, const MeasParams* params=nullptr) { 
        if (!params) {
            throw exceptions::BadParams("Range and Bearing requires parameters.");
        }
        if (!utilities::instanceof<RangeAndBearingParams>(params)) {
            throw exceptions::BadParams("params type must be RangeAndBearingParams.");
        }
        auto ptr = dynamic_cast<const RangeAndBearingParams*>(params);

        return atan2(state(ptr->yInd), state(ptr->xInd));
    }
};
} // namespace lager::gncpy::measurements