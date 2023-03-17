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
    lager::gncpy::matrix::Matrix<float> m ({{2, -1, -2.},{-4, 6, 3},{-4, -2, 8}});
    float a = 2;
    lager::gncpy::matrix::Matrix<float> b = a * m;
    std::cout<<b;
}