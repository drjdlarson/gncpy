cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME gncpy)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for sonarlint plugin/sonarqube

file(GLOB_RECURSE SRCS src/*.cpp)
file(GLOB_RECURSE HDRS include/*.h)

project(${PROJECT_NAME})


add_library(${PROJECT_NAME} ${SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC include)