cmake_minimum_required(VERSION 3.13)

# Project information
project(Gnss
	VERSION 2.4.0
	DESCRIPTION "GNSS interface"
	LANGUAGES CXX
)

# Add the library target
add_library(gnss INTERFACE)

# Setup include directories 
target_include_directories(gnss INTERFACE include/)
