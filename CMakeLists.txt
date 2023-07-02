cmake_minimum_required(VERSION 3.25)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/support/cmake"
)

# Import `CMakeExtraUtils` or bundle `DynamicVersion.cmake` from there
cmake_policy(SET CMP0140 NEW)
include(DynamicVersion)
include(CompileOptions)

# Set ${PROJECT_VERSION} according to git tag or `.git_archival.txt`
dynamic_version(PROJECT_PREFIX GNCPY_CPP_)

project(${SKBUILD_PROJECT_NAME} VERSION ${PROJECT_VERSION})

option(GNCPY_CPP_VERSION "Version of the c++ backend (without the v prefix) for release builds")
set(GNCPY_CPP_BRANCH "master" CACHE STRING "Branch of the c++ backend for debug builds")
set(GNCPY_EIGEN_VERSION "3.4.0" CACHE STRING "Version of Eigen to use when compiling the interface and backend")

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)

message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Wrapping GNCPy version: ${PROJECT_VERSION}")



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

