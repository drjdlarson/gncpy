cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

option(GNCPY_CPP_VERSION "Version of the c++ backend (without the v prefix) for release builds")
set(GNCPY_CPP_BRANCH "master" CACHE STRING "Branch of the c++ backend for debug builds")
set(GNCPY_EIGEN_VERSION "3.4.0" CACHE STRING "Version of Eigen to use when compiling the interface and backend")

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
message(STATUS "Wrapping GNCPy version: ${SKBUILD_PROJECT_VERSION}")



# -----------------------------------------------------------------
# -------------------- C++ dependency stuff -----------------------
# -----------------------------------------------------------------
include(FetchContent)
if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    message(STATUS "Fetching gncpy_cpp (version: ${GNCPY_CPP_VERSION})...")
    FetchContent_Declare(
        gncpy_cpp
        GIT_REPOSITORY https://github.com/drjdlarson/gncpy_cpp.git
        GIT_TAG "v${GNCPY_CPP_VERSION}"
    )
else()
message(STATUS "Fetching gncpy_cpp (branch: ${GNCPY_CPP_BRANCH})...")
    FetchContent_Declare(
        gncpy_cpp
        GIT_REPOSITORY https://github.com/drjdlarson/gncpy_cpp.git
        GIT_TAG ${GNCPY_CPP_BRANCH}
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

message(STATUS "Fetching Eigen3 dependency...")
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG ${GNCPY_EIGEN_VERSION}
)
set(BUILD_TESTING OFF CACHE INTERNAL "Disable Eigen tests")
FetchContent_MakeAvailable(eigen)


# -----------------------------------------------------------------
# ------------------- create wrapper modules ----------------------
# -----------------------------------------------------------------
add_subdirectory(interface)
