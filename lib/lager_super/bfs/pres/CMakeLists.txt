cmake_minimum_required(VERSION 3.13)

# Project information
project(Pres
	VERSION 1.2.0
	DESCRIPTION "Pressure interface"
	LANGUAGES CXX
)

# Add the library target
add_library(pres INTERFACE)

# Setup include directories 
target_include_directories(pres INTERFACE include/)
