cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

option(GNCPY_CPP_VERSION "Version of the c++ backend (without the v prefix)" 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra -Wpedantic")

if(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug")
    set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4 -gstatement-frontiers -gvariable-location-views -ginline-points -fno-eliminate-unused-debug-symbols -fvar-tracking -fvar-tracking-assignments -Og")

elseif(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

endif()

message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using GNCPy-c++ version: v${GNCPY_CPP_VERSION}")
message(STATUS "Using GNCPy version: ${SKBUILD_PROJECT_VERSION}")


# -----------------------------------------------------------------
# -------------------- C++ dependency stuff -----------------------
# -----------------------------------------------------------------
include(FetchContent)
if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    FetchContent_Declare(
        gncpy_cpp
        GIT_REPOSITORY https://github.com/ryan4984/gncpy_cpp.git
        GIT_TAG "v${GNCPY_CPP_VERSION}"
    )
# TODO: figure out how to clone branch of repo for debug builds (following fails to generate valid tar file)
else()
    # TODO: make sure this will work to get latest on master branch
    # TODO: add option to get other branches
    FetchContent_Declare(
        gncpy_cpp
        GIT_REPOSITORY https://github.com/ryan4984/gncpy_cpp
        GIT_TAG master
    )
endif()
set(GNCPY_DOC OFF CACHE INTERNAL "Skip building docs")
set(GNCPY_TEST OFF CACHE INTERNAL "Skip building tests")
set(GNCPY_INSTALL ON CACHE INTERNAL "Generate the install target")
FetchContent_MakeAvailable(gncpy_cpp)


# -----------------------------------------------------------------
# ----------------- wrapper dependency stuff ----------------------
# -----------------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)


# -----------------------------------------------------------------
# ------------------- create wrapper modules ----------------------
# -----------------------------------------------------------------
add_subdirectory(interface)
