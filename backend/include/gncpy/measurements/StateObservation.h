#pragma once
#include <functional>
#include <vector>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Math.h"
#include "gncpy/measurements/Parameters.h"
#include "gncpy/measurements/ILinearMeasModel.h"
#include "gncpy/Utilities.h"

namespace lager::gncpy::measurements {

class StateObservationParams : public MeasParams {
public:
    std::vector<uint8_t> obsInds;
    StateObservationParams(const std::vector<uint8_t> obsInds)
    : obsInds(obsInds) {

    }
};

template<typename T>
class StateObservation : public ILinearMeasModel<T> {
public:
    matrix::Matrix<T> getMeasMat(const matrix::Vector<T>& state, const std::unique_ptr<MeasParams>& params=std::make_unique<MeasParams>()) const override {
        if (!params) {
            throw BadParams("State Observation requires parameters");
        }
        
        auto ptr = dynamic_cast<StateObservationParams*>(params.get());
        matrix::Matrix<T> data(ptr->obsInds.size(), state.size());

        for (uint8_t ii=0;ii<ptr->obsInds.size();ii++) {
            for (uint8_t jj=0;jj<state.size();jj++) {
                if (ptr->obsInds[ii] == jj) {
                    data(ii, jj) = 1.0;
                }
                else {
                    data(ii, jj) = 0.0;
                }
            }
        }
        return data;
    }    
};
}  // namespace lager::gncpy::measurement
