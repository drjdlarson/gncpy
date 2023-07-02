pybind11_add_module(_measurements MODULE 
    Initialize.cpp
    Interface.cpp
    Parameters.cpp
    StateObservation.cpp
    RangeAndBearing.cpp
    Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Macros.h
)
target_compile_definitions(_measurements PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_measurements
    PRIVATE
        Eigen3::Eigen
        lager::gncpy
)
compiler_options(COMPILE_TARGET _measurements)
install(TARGETS _measurements 
    DESTINATION 
        ${SKBUILD_PROJECT_NAME}/measurements
)