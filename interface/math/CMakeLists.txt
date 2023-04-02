#---------------------------------------------------------------------------
# setup Matrix module
#---------------------------------------------------------------------------
pybind11_add_module(_matrix MODULE 
    Matrix.cpp
    Common.h
)
target_compile_definitions(_matrix PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(_matrix
    PRIVATE
        lager::gncpy
)
install(TARGETS _matrix
    DESTINATION 
        gncpy/math
)