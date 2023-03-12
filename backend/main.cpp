#include "gncpy/math/Matrix.h"
#include <iostream>
#include <stdint.h>

int main() {
    lager::gncpy::matrix::Matrix<float> m ({{10.69, 20.5, 40.},{30.0000001, 40.2222222, 50}});
    std::cout<< m;
    std::cout<<"\n";
    lager::gncpy::matrix::Matrix<float> n = m.transpose();
    std::cout<< n;
}