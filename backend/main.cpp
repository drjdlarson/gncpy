#include <iostream>
#include "gncpy/dynamics/DoubleIntegrator.h"


int main() {
    lager::gncpy::dynamics::DoubleIntegrator<double> dyn(0.1);

    return 0;
}