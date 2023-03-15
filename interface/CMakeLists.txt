pybind11_add_module(_dynamics MODULE Dynamics.cpp)
target_compile_definitions(_dynamics PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_dynamics
    PRIVATE
        lager::gncpy_backend
)

message("INTERFACE CURRENT INSTALL DIR: ${SKBUILD_DIR}")
install(TARGETS _dynamics 
    DESTINATION 
        gncpy/dynamics
)