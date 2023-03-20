#pragma once
#include <cmath>
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
matrix::Matrix<T> getJacobian(const matrix::Vector<T>& x, const std::vector<std::function<T (const lager::gncpy::matrix::Vector<T>&)>>& fncLst) {
    std::vector<T> data;
    for(auto const& f : fncLst) {
        for(auto & val : getJacobian(x, f)) {
            data.emplace_back(val);
        }
    }

    return matrix::Matrix(fncLst.size(), x.size(), data);
}

template<typename T>
T calcGaussianPDF(const matrix::Vector<T>& x, const matrix::Vector<T>& m, const matrix::Matrix<T>& cov) {
    uint8_t nDim = x.size();
    T val;
    if(nDim > 1) {
        matrix::Vector<T> diff = x - m;
        val = -0.5*(static_cast<T>(nDim) * std::log(static_cast<T>(2) * M_PI) + std::log(cov.determinant()) + (diff.transpose() * cov.inverse() * diff).toScalar());
    } else {
        T diff = (x - m).toScalar();
        val = -0.5 * (std::log(static_cast<T>(2) * M_PI * cov.toScalar()) + std::pow(diff, 2) / cov.toScalar());
    }
    return std::exp(val);
}

} // namespace lager::gncpy::math
