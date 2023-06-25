On adding new filters to this repository:

Must declare the init*FilterName* function in Common.h

Must include a *FilterName*.cpp file including the pybind wrapper code for the init*FilterName* function.

Must include init*FilterName*(m) inside of Initialize.cpp