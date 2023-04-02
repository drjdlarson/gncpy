#include <iostream>
#include <gncpy/dynamics/DoubleIntegrator.h>

int main() {
    lager::gncpy::dynamics::DoubleIntegrator<double> obj(0.5);
    std::cout << "Address of double integrator " << &obj << std::endl;

    return 0;
}
