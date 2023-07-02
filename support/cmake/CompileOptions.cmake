function (compiler_options)


    set(ARGS_Options "")
    set(ARGS_OneValue "")
    set(ARGS_MultiValue "")
    list(APPEND ARGS_OneValue
            COMPILE_TARGET
        )

    cmake_parse_arguments(ARGS "${ARGS_Options}" "${ARGS_OneValue}" "${ARGS_MultiValue}" ${ARGN})

    if(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            message(STATUS "Using GNU compiler, adding full set of compiler flags")
            target_compile_options(${COMPILE_TARGET} PRIVATE
                "-fPIC"
                "-Wall"
                "-Wextra"
                "-Wpedantic"
                "-gdwarf-4"
                "-gstatement-frontiers"
                "-gvariable-location-views"
                "-ginline-points"
                "-fno-eliminate-unused-debug-symbols"
                "-fvar-tracking"
                "-fvar-tracking-assignments"
            )
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            message(STATUS "Using MSVC compiler, adding reduced set of compiler flags")
                target_compile_options(${COMPILE_TARGET} PRIVATE
                "-Wall"
                "-bigobj"
                "-Od"
            )
        else()
            message(STATUS "Not using GNU or MSVC compiler, adding reduced set of compiler flags")
                target_compile_options(${COMPILE_TARGET} PRIVATE
                "-fPIC"
                "-Wall"
                "-Wextra"
                "-Wpedantic"
                "-gdwarf-4"
                "-Og"
            )
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            message(STATUS "Using MSVC compiler, adding reduced set of compiler flags")
                target_compile_options(${COMPILE_TARGET} PRIVATE
                "-Wall"
                "-bigobj"
                "-O3"
            )
        else()
            message(STATUS "Not using MSVC compiler, adding full set of compiler flags")
            target_compile_options(${COMPILE_TARGET} PRIVATE
                "-fPIC"
                "-Wall"
                "-Wextra"
                "-Wpedantic"
                "-O3"
            )
        endif()
    endif()
endfunction()