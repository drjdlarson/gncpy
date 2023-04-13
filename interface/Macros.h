#pragma once

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// see https://github.com/pybind/pybind11/issues/956 for why the shared_ptr is needed
#define GNCPY_PY_CHILD_CLASS(child, base) py::class_<child, base, std::shared_ptr<child>>
#define GNCPY_PY_BASE_CLASS(base) py::class_<base, std::shared_ptr<base>>

#define GNCPY_STR(class_t) \
    .def("__str__", [](const class_t& self) { \
                return self.toJSON(); \
    }) \
    .def("__repr__", [](const class_t& self) { \
            return self.toJSON(); \
        } \
    )

#define GNCPY_PICKLE(class_t) \
    .def(py::pickle( \
            [](const class_t& self) { /* __getstate__ */ \
                return py::make_tuple(py::bytes(self.saveClassState().str())); \
            }, \
            [](py::tuple t) { /* __setstate__ */ \
                if(t.size() != 1) { \
                    throw std::runtime_error("Invalid state!"); \
                } \
                std::stringstream ssb(t[0].cast<std::string>(), std::ios::in | std::ios::out | std::ios::binary); \
                auto obj = class_t::loadClass(ssb); \
                std::cout << "in c++:" << obj.toJSON() << std::endl; \
                return obj; \
            } \
        )) \
        GNCPY_STR(class_t)
