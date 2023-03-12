cmake_minimum_required(VERSION 3.10)
set(MY_PROJECT gncpy)

#---------------------------------------------------------------------------
# cmake options
#---------------------------------------------------------------------------
option(ENABLE_TESTING "Build the tests" ON)
option(BUILD_DOC "Build the documentation" ON)


#---------------------------------------------------------------------------
# library metadata
#---------------------------------------------------------------------------
project(${MY_PROJECT}
        VERSION 0.1
        DESCRIPTION "Backend for the GNCPy python package"
        HOMEPAGE_URL "https://github.com/drjdlarson/gncpy"
        LANGUAGES CXX
)

#---------------------------------------------------------------------------
# compile settings
#---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for sonarlint plugin/sonarqube
set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra -Wpedantic")

if(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug")
    message("DEBUG mode")
    set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4 -gstatement-frontiers -gvariable-location-views -ginline-points -fno-eliminate-unused-debug-symbols -fvar-tracking -fvar-tracking-assignments -Og")

elseif(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    message("RELEASE mode")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

endif()


#---------------------------------------------------------------------------
# input files
#---------------------------------------------------------------------------
file(GLOB_RECURSE SRCS src/*.cpp)
file(GLOB_RECURSE HDRS include/*.h)


#---------------------------------------------------------------------------
# check for testing code
#---------------------------------------------------------------------------
if(ENABLE_TESTING)
    message("Enabling testing...")
    enable_testing()
    add_subdirectory(tests)
endif(ENABLE_TESTING)


#---------------------------------------------------------------------------
# check for building documentation
#---------------------------------------------------------------------------
# see: https://vicrucann.github.io/tutorials/quick-cmake-doxygen/
if(BUILD_DOC)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        # request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message("Doxygen build started")

        # note the option ALL which allows to build the docs together with the application
        add_custom_target(doc_doxygen ALL
                          COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                          COMMENT "Generating API documentation with Doxygen"
                          VERBATIM
        )
    else()
        message("Doxygen needs to be installed to generate the doxygen documentation")
    endif()
endif()




#---------------------------------------------------------------------------
# setup library
#---------------------------------------------------------------------------
add_library(${MY_PROJECT} ${SRCS} ${HDRS})
target_include_directories(${MY_PROJECT} PUBLIC include)
add_executable(main main.cpp)
target_link_libraries(main ${MY_PROJECT})
target_include_directories(main PUBLIC include)
