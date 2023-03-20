#include <iostream>
#include <stdint.h>
#include <memory>
#include <optional>
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Vector.h"
#include "gncpy/filters/Kalman.h"
#include "gncpy/filters/Parameters.h"
#include "gncpy/dynamics/Parameters.h"

int main() {
    lager::gncpy::matrix::Matrix<float> m ({{1,2,3,4},{6,7,8,9},{10,11,12,13},{14,15,16,17}});
    lager::gncpy::matrix::Matrix<float> n ({{99,98,97},{96,95,94},{93,92,91}});
    m(0,0,3,3,n);
    std::cout<<m;
}
