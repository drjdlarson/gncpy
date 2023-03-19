#pragma once
#include <vector>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Math.h"
#include "gncpy/Exceptions.h"
#include "gncpy/measurements/Parameters.h"
#include "gncpy/measurements/ILinearMeasModel.h"
#include "gncpy/Utilities.h"

namespace lager::gncpy::measurements {

class StateObservationParams final: public MeasParams {
public:
    explicit StateObservationParams(const std::vector<uint8_t>& obsInds)
    : obsInds(obsInds) {

    }

    std::vector<uint8_t> obsInds;
};


template<typename T>
class StateObservation final: public ILinearMeasModel<T> {
public:
    matrix::Matrix<T> getMeasMat(const matrix::Vector<T>& state, const MeasParams* params=nullptr) const override {
        if (!params) {
            throw exceptions::BadParams("State Observation requires parameters");
        }
        if (!utilities::instanceof<StateObservationParams>(params)) {
            throw exceptions::BadParams("params type must be StateObservationParams.");
        }
        auto ptr = dynamic_cast<const StateObservationParams*>(params);
        matrix::Matrix<T> data(ptr->obsInds.size(), state.size());

        for(uint8_t ii = 0; ii < ptr->obsInds.size(); ii++) {
            for (uint8_t jj = 0; jj < state.size(); jj++) {
                if (ptr->obsInds[ii] == jj) {
                    data(ii, jj) = static_cast<T>(1.0);
                }
            }
        }
        return data;
    }

};
}  // namespace lager::gncpy::measurement
