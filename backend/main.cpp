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
    lager::gncpy::matrix::Matrix<float> m ({{2, -1, -2., 3, 4, 5, 7, 8},{-4, 6, 3, 4, 6, 1, 2, 9},
                                            {-4, -2, 8, 5, 6, 7, 8, 21},{14, 2, 2, 5, 3, 6, 3, 2},
                                            {-3, -7, 10, 2, 1, 3, 4, 2},{1, 6, 23, 54, 0.1, 0, 1, 3},
                                            {2, 3, 4, 1, 6, 7, 3, 0.5},{4, 2, 0.1, 7, 4, 2, 51, 3}});
    lager::gncpy::matrix::Matrix<float> n = m.inverse();
    std::cout<<m<<"\n"<<n;
}