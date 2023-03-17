cmake_minimum_required(VERSION 3.10)
set(BACKEND_PROJECT gncpy_backend)
set(GNCPY_BACKEND_VERSION 0.0.0)

#---------------------------------------------------------------------------
# cmake options
#---------------------------------------------------------------------------
option(ENABLE_TESTING "Build the tests" ON)
option(BUILD_DOC "Build the documentation" ON)


#---------------------------------------------------------------------------
# library metadata
#---------------------------------------------------------------------------
project(${BACKEND_PROJECT} LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

#---------------------------------------------------------------------------
# compile settings
#---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for sonarlint plugin/sonarqube
set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)


set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra -Wpedantic")

if(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug")
    message("DEBUG mode")
    set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4 -gstatement-frontiers -gvariable-location-views -ginline-points -fno-eliminate-unused-debug-symbols -fvar-tracking -fvar-tracking-assignments -Og")

elseif(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    message("RELEASE mode")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

endif()


set(cmake_package_name lager CACHE INTERNAL "")


#---------------------------------------------------------------------------
# input files
#---------------------------------------------------------------------------
file(GLOB_RECURSE SRCS src/*.cpp)
file(GLOB_RECURSE HDRS include/*.h)


#---------------------------------------------------------------------------
# setup library
#---------------------------------------------------------------------------
# see: https://stackoverflow.com/questions/55635294/how-to-create-packages-with-cmake for details/help
# and https://stackoverflow.com/questions/20746936/what-use-is-find-package-when-you-need-to-specify-cmake-module-path

# add library and set properties
add_library(${BACKEND_PROJECT} "" ${HDRS} ${SRCS})
add_library(lager::gncpy_backend ALIAS ${BACKEND_PROJECT})

set(gncpy_backend_build_include_dirs
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${gncpy_backend_build_include_dirs})

set_target_properties(${BACKEND_PROJECT} 
    PROPERTIES
        VERSION ${GNCPY_BACKEND_VERSION}
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

string(REPLACE ";" "$<SEMICOLON>" dirs "${gncpy_backend_build_include_dirs}")
target_include_directories(${BACKEND_PROJECT} SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${dirs}>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>")


# setup install and declare export for the target
install(TARGETS ${BACKEND_PROJECT}
    EXPORT BackendTargets
    ARCHIVE DESTINATION ${CMAKE_ISNTALL_LIBDIR}
)


# setup install for the exports
install(
    EXPORT BackendTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lager
    NAMESPACE lager::
    FILE BackendTargets.cmake
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
    "lager::gncpy_backendConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lager
    PATH_VARS
        CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/lager::gncpy_backendConfigVersion.cmake
    VERSION ${GNCPY_BACKEND_VERSION}
    COMPATIBILITY SameMajorVersion
)


install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/lager::gncpy_backendConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/lager::gncpy_backendConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lager"
)












# old stuff
# ----------------------------------------------------------------------------




# Set the output directory for build artifacts
# set_target_properties(${BACKEND_PROJECT} 
#     PROPERTIES
#         VERSION ${GNCPY_BACKEND_VERSION}
#         RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#         LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
#         ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
#         PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#         COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
# )
# # make PDBs match library name
# get_target_property(pdb_debug_postfix ${BACKEND_PROJECT} DEBUG_POSTFIX)
# set_target_properties(${BACKEND_PROJECT}
#     PROPERTIES
#         PDB_NAME "${BACKEND_PROJECT}"
#         PDB_NAME_DEBUG "${BACKEND_PROJECT}${pdb_debug_postfix}"
#         COMPILE_PDB_NAME "${BACKEND_PROJECT}"
#         COMPILE_PDB_NAME_DEBUG "${BACKEND_PROJECT}${pdb_debug_postfix}"
# )

# string(REPLACE ";" "$<SEMICOLON>" dirs "${gtest_build_include_dirs}")
# target_include_directories(${BACKEND_PROJECT} SYSTEM INTERFACE
#     "$<BUILD_INTERFACE:${dirs}>"
#     "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>")
# install(TARGETS ${BACKEND_PROJECT}
#     EXPORT ${targets_export_name}
#     RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
#     ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#     LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
# )
# set(configured_pc "${generated_dir}/${BACKEND_PROJECT}.pc")
# configure_file("${PROJECT_SOURCE_DIR}/cmake/${BACKEND_PROJECT}.pc.in"
#     "${configured_pc}" @ONLY
# )
# install(FILES "${configured_pc}" DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")



#---------------------------------------------------------------------------
# check for testing code
#---------------------------------------------------------------------------
if(ENABLE_TESTING)
    message("Enabling testing...")
    enable_testing()
    add_subdirectory(tests)
endif(ENABLE_TESTING)


#---------------------------------------------------------------------------
# check for building documentation
#---------------------------------------------------------------------------
# see: https://vicrucann.github.io/tutorials/quick-cmake-doxygen/
if(BUILD_DOC)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        # request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message("Doxygen build started")

        # note the option ALL which allows to build the docs together with the application
        add_custom_target(doc_doxygen ALL
                          COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                          COMMENT "Generating API documentation with Doxygen"
                          VERBATIM
        )
    else()
        message("Doxygen needs to be installed to generate the doxygen documentation")
    endif()
endif()


#---------------------------------------------------------------------------
# temp for debugging
#---------------------------------------------------------------------------
add_executable(main main.cpp)
target_link_libraries(main lager::gncpy_backend)
target_include_directories(main PUBLIC include)

# add_executable(example_kf example_kf.cpp)
# target_link_libraries(example_kf ${BACKEND_PROJECT})
# target_include_directories(example_kf PUBLIC include)
