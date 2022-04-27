cmake_minimum_required(VERSION 3.0)

set(TARGET lager_super_bindings)

project(${TARGET})

add_subdirectory(lib/lager_super)
add_subdirectory(lib/pybind11)

set(SOURCE_DIR "gncpy/dynamics/aircraft")
set(SOURCES "${SOURCE_DIR}/lager_super_bindings.cpp")

pybind11_add_module(${TARGET} MODULE NO_EXTRAS ${SOURCES})

target_compile_options(${TARGET} PRIVATE
    -std=c++2a
    -fconcepts
    -felide-constructors
    -Wno-error=narrowing
    -Wno-volatile
    -Wno-psabi
    -ffunction-sections
    -fdata-sections
    -MMD
)

target_link_libraries(${TARGET}
    PRIVATE
        lager_super
)
