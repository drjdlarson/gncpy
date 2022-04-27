cmake_minimum_required(VERSION 3.0)

set(TARGET lager_queso_bindings)

project(${TARGET})

#add_definitions(-D__FMU_R_V2__)

add_subdirectory(lib/lager_queso)
add_subdirectory(lib/pybind11)

set(SOURCE_DIR "gncpy/dynamics/aircraft")
set(SOURCES "${SOURCE_DIR}/lager_queso_bindings.cpp")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -fconcepts -felide-constructors -Wno-error=narrowing -Wno-volatile -g -Os -Wno-psabi -ffunction-sections -fdata-sections -nostdlib -MMD")
# set(CMAKE_C_FLAGS, "${CMAKE_C_FLAGS} -std=c11")

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
        lager_queso
)
