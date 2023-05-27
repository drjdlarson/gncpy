# see https://stackoverflow.com/questions/53762552/with-pybind11-how-to-split-my-code-into-multiple-modules-files for multiple directory info

#---------------------------------------------------------------------------
# setup Dynamics directory
#---------------------------------------------------------------------------
add_subdirectory(dynamics)

#---------------------------------------------------------------------------
# setup measurements directory
#---------------------------------------------------------------------------
add_subdirectory(measurements)

#---------------------------------------------------------------------------
# setup filters directory
#---------------------------------------------------------------------------
add_subdirectory(filters)