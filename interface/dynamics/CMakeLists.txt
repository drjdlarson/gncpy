pybind11_add_module(_dynamics MODULE 
    Initialize.cpp
    Interface.cpp
    Parameters.cpp
    DoubleIntegrator.cpp
    Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Macros.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../math/Common.h
)
target_compile_definitions(_dynamics PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_dynamics
    PRIVATE
        lager::gncpy
)
install(TARGETS _dynamics 
    DESTINATION 
        gncpy/dynamics
)