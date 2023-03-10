#include <iostream>
#include "dynamics/DoubleIntegrator.h"


int main() {
    lager::gncpy::dynamics::DoubleIntegrator<double> dyn(0.1);

    return 0;
}