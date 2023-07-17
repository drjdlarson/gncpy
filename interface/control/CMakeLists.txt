pybind11_add_module(_control MODULE 
    Initialize.cpp
    Interface.cpp
    Parameters.cpp
    StateControl.cpp
    Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Macros.h
)
target_compile_definitions(_control PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_control
    PRIVATE
        Eigen3::Eigen
        lager::gncpy
)
compiler_options(COMPILE_TARGET _control)
install(TARGETS _control
    DESTINATION 
        ${SKBUILD_PROJECT_NAME}/control
)