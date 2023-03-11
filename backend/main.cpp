#include <iostream>
#include <sys/resource.h>
#include "gncpy/dynamics/DoubleIntegrator.h"
#include "gncpy/math/Vector.h"


int main() {
    double dt = 0.1;
    lager::gncpy::dynamics::DoubleIntegrator dyn(dt);
    lager::gncpy::matrix::Vector xk({0., 0., 1., 0.});
    lager::gncpy::matrix::Vector xk1 = xk;

    for(uint16_t kk = 0; kk < 10; kk++) {
        double timestep = kk * dt;
        std::cout << "t = " << timestep << ": ";

        xk1 = dyn.propagateState(timestep, xk);
        for(auto const& x : xk1) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        xk = xk1;
    }


    return 0;
}