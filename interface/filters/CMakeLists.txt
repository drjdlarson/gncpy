pybind11_add_module(_filters MODULE 
    Initialize.cpp
    Interface.cpp
    Parameters.cpp
    Kalman.cpp
    Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Macros.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../math/Common.h
)
target_compile_definitions(_filters PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_filters
    PRIVATE
        lager::gncpy_backend
)
install(TARGETS _filters 
    DESTINATION 
        gncpy/filters
)