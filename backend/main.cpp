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
    lager::gncpy::matrix::Vector<float> v ({1,2,3});
    lager::gncpy::matrix::Vector<float> u ({4,5,6});
    float b = v.dot(u);
    lager::gncpy::matrix::Vector<float> a = v.cross(u);
    std::cout<<b;
}
