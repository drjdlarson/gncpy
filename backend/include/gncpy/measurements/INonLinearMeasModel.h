#pragma once
#include <functional>
#include <vector>
#include "gncpy/math/Vector.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Math.h"
#include "gncpy/measurements/Parameters.h"
#include "gncpy/measurements/IMeasModel.h"



namespace lager::gncpy::measurements{

template<typename T>
class INonLinearMeasModel : public IMeasModel<T> {
public:
    matrix::Vector<T> measure(const matrix::Vector<T>& state, const MeasParams* params=nullptr) const override {
        std::vector<T> data;
        for (auto const& h : this->getMeasFuncLst(params)) {
            data.emplace_back(h(state));
        }
        return matrix::Vector<T>(data.size(), data);
    }
    matrix::Matrix<T> getMeasMat(const matrix::Vector<T>& state, const MeasParams* params=nullptr) const override {
        return math::getJacobian(state, this->getMeasFuncLst(params));
    };
protected:
    virtual std::vector<std::function<T (const matrix::Vector<T>&)>> getMeasFuncLst(const MeasParams* params=nullptr) const = 0; 

};
} // namespace lager::gncpy::measurements