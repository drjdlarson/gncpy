cmake_minimum_required(VERSION 3.13)

set(TARGET lager_super)

project(${TARGET} VERSION 0.1)
message(STATUS "Building ${TARGET}")

set(FMU v2)
set(AUTOCODE baseline_super)

message(STATUS "Using autocode: ${AUTOCODE}")
message(STATUS "Using fmu: ${FMU}")

add_library(${TARGET} STATIC ${AUTOCODE}_ert_rtw/autocode.cpp)

target_compile_options(${TARGET} PRIVATE
    -std=c++2a
    -fconcepts
    -felide-constructors
    -Wno-error=narrowing
    -Wno-volatile
    -g
    -Wno-psabi
    -ffunction-sections
    -fdata-sections
    -MMD
    -fPIC
)


#################### from bfs make files ######################
# FMU version
if (DEFINED FMU)
	string(TOUPPER ${FMU} FMU)
endif()
if (FMU STREQUAL "V2")
    # FMU-R-V2
    set(FMU_DEF __FMU_R_V2__)
elseif(FMU STREQUAL "V2-BETA")
    # FMU-R-V2-BETA
    set(FMU_DEF __FMU_R_V2__)
else()
	# FMU-R-V1
    set(FMU_DEF __FMU_R_V1__)
endif()

# Fetch dependencies
add_subdirectory(bfs/global_defs)
add_subdirectory(bfs/imu)
add_subdirectory(bfs/gnss)
add_subdirectory(bfs/pres)

# Setup autocode
target_compile_definitions(${TARGET} PUBLIC
    __AUTOCODE__
    ${FMU_DEF}
)

# Add the includes
target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/bfs/global_defs/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bfs/imu/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bfs/gnss/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bfs/pres/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${AUTOCODE}_ert_rtw
)

# Link libraries to the executable
target_link_libraries(${TARGET}
    PUBLIC
        global_defs
        imu
        gnss
        pres
)
