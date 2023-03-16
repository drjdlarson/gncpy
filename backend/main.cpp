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
    lager::gncpy::matrix::Matrix<float> m ({{10.69, 20.5, 40.},{30.0000001, 40.2222222, 50}});
    bool transposed = m.beenTransposed();
    std::cout<< m;
    std::cout<<"\n";
    m.transpose(true);
    transposed = m.beenTransposed();
    std::cout << m;
}