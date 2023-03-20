#pragma once

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// see https://github.com/pybind/pybind11/issues/956 for why the shared_ptr is needed
#define GNCPY_PY_CHILD_CLASS(child, base) py::class_<child, base, std::shared_ptr<child>>
#define GNCPY_PY_BASE_CLASS(base) py::class_<base, std::shared_ptr<base>>