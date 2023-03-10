cmake_minimum_required(VERSION 3.10)
set(MY_PROJECT gncpy)

#---------------------------------------------------------------------------
# cmake options
#---------------------------------------------------------------------------
option(ENABLE_TESTING "Build the tests" ON)


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
set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra -Wpedantic -O3")


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
# setup library
#---------------------------------------------------------------------------
add_library(${MY_PROJECT} ${SRCS} ${HDRS})
target_include_directories(${MY_PROJECT} PUBLIC include)


#---------------------------------------------------------------------------
# temp for debugging
#---------------------------------------------------------------------------
add_executable(test_main main.cpp)
target_link_libraries(test_main ${MY_PROJECT})
target_include_directories(test_main PUBLIC include)