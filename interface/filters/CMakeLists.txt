pybind11_add_module(_filters MODULE 
    Initialize.cpp
    Interface.cpp
    Parameters.cpp
    Kalman.cpp
    ExtendedKalman.cpp
    Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Macros.h
)
target_compile_definitions(_filters PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_filters
    PRIVATE
        Eigen3::Eigen
        lager::gncpy
)
compiler_options(COMPILE_TARGET _filters)
install(TARGETS _filters 
    DESTINATION 
        ${SKBUILD_PROJECT_NAME}/filters
)