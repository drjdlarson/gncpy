pybind11_add_module(_dynamics MODULE 
    Initialize.cpp
    Interface.cpp
    Parameters.cpp
    DoubleIntegrator.cpp
    ClohessyWiltshire.cpp
    Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Macros.h
)
target_compile_definitions(_dynamics PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_dynamics
    PRIVATE
        Eigen3::Eigen
        lager::gncpy
)
compiler_options(COMPILE_TARGET _dynamics)
install(TARGETS _dynamics 
    DESTINATION 
        ${SKBUILD_PROJECT_NAME}/dynamics
)