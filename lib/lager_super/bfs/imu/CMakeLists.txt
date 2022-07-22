cmake_minimum_required(VERSION 3.13)
# Project information
project(Imu
	VERSION 3.0.0
	DESCRIPTION "IMU interface"
	LANGUAGES CXX
)
# Add the library target
add_library(imu INTERFACE)
# Setup include directories 
target_include_directories(imu INTERFACE include/)
# Compile options
add_compile_options(
	$<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
)
