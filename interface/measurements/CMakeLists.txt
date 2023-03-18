pybind11_add_module(_measurements MODULE 
    Initialize.cpp
    Interface.cpp
    Parameters.cpp
    StateObservation.cpp
    Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Macros.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../math/Common.h
)
target_compile_definitions(_measurements PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_measurements
    PRIVATE
        lager::gncpy_backend
)
install(TARGETS _measurements 
    DESTINATION 
        gncpy/measurements
)