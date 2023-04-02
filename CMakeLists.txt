cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(GNCPY_CPP_VERSION 0.1.0)


# -----------------------------------------------------------------
# ------------------ conan dependency stuff -----------------------
# -----------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_BUILD_TYPE Release)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

# declare the dependencies of the wrapping code (ie the c++ backend)
conan_cmake_configure(
    REQUIRES
        gncpy/${GNCPY_CPP_VERSION}
    GENERATORS 
        cmake_find_package
)

# populate conan command and create default profile (required by the remaining commands)
conan_check(REQUIRED)

# set the conan home folder to the local pip folder to make debugging easier and avoid overwriting any existing cache in user space
set(ENV{CONAN_USER_HOME} ${CMAKE_CURRENT_LIST_DIR}) # conan v1
set(ENV{CONAN_HOME} ${CMAKE_CURRENT_LIST_DIR}) # conan v2
message(STATUS "conan v1 home dir: '" $ENV{CONAN_USER_HOME} "'")
message(STATUS "conan v2 home dir: '" $ENV{CONAN_HOME} "'")

# translate cmake settings into conan cmd line args
conan_cmake_autodetect(settings)

# TODO: this is for local builds, full deploy should use conan center
# mimic conan create but with more control over the install process since binary does not exist in conan center for local/dev builds
message(STATUS "Creating dev package for local builds")
execute_process(COMMAND ${CONAN_CMD} export ${CMAKE_CURRENT_LIST_DIR}/recipe lager/testing
                    RESULT_VARIABLE return_code
                    ${OUTPUT_OPT}
                    ${ERROR_OPT}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
conan_cmake_install(PATH_OR_REFERENCE gncpy/${GNCPY_CPP_VERSION}@lager/testing
                    BUILD gncpy
                    GENERATOR deploy
                    SETTINGS ${settings}
                    INSTALL_FOLDER ${CMAKE_CURRENT_BINARY_DIR}
)

# TODO: in full deploy form conan center should not need path hint
find_package(gncpy CONFIG REQUIRED
            PATHS ${CMAKE_CURRENT_BINARY_DIR}
)


# -----------------------------------------------------------------
# ----------------- wrapper dependency stuff ----------------------
# -----------------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(interface)
