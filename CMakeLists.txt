cmake_minimum_required(VERSION 3.15)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

option(ENABLE_TESTING OFF)
set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(backend)
add_subdirectory(interface)

