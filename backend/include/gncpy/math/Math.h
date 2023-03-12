#pragma once
#include <vector>
#include <functional>
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Vector.h"


namespace lager::gncpy::math {

template<typename T, typename F>
matrix::Vector<T> getJacobian(const matrix::Vector<T>& x, const F& fnc) {
    const double step = 1e-7;
    const T invStep2 = 1. / (2. * step);
    
    std::vector<T> data;
    matrix::Vector xR(x);
    matrix::Vector xL(x);
    for(uint8_t ii = 0; ii < x.size(); ii++) {
        xR(ii) += step;
        xL(ii) -= step;

        data.emplace_back((fnc(xR) - fnc(xL)) * invStep2);

        // reset for next round
        xR(ii) -= step;
        xL(ii) += step;
    }

    return matrix::Vector(data.size(), data);
}


template<typename T>
matrix::Matrix<T> getJacobian(const matrix::Vector<T>& x, const std::vector<std::function<T (lager::gncpy::matrix::Vector<T>&)>>& fncLst) {
    std::vector<T> data;
    for(auto const& f : fncLst) {
        for(auto & val : getJacobian(x, f)) {
            data.emplace_back(val);
        }
    }

    return matrix::Matrix(fncLst.size(), x.size(), data);
}

} // namespace lager::gncpy::math
