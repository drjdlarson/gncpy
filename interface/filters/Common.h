#pragma once
#include <pybind11/pybind11.h>

extern void initInterface(pybind11::module&);
extern void initParameters(pybind11::module&);
extern void initKalman(pybind11::module&);