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
    lager::gncpy::matrix::Matrix<float> l = lager::gncpy::matrix::Matrix<float>(3,3);
    lager::gncpy::matrix::Matrix<float> u = l;
    lager::gncpy::matrix::Matrix<float> I = lager::gncpy::matrix::identity<float>(3);
    
    m.LU_decomp(l,u);
    //std::cout<<m;
    //std::cout<<"\n";
    //std::cout<<l;
    //std::cout<<"\n";
    //std::cout<<u;
    //std::cout<<"\n";
    lager::gncpy::matrix::Matrix<float> n = m.inverse();
    //std::cout<<n;

}