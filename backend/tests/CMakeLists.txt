# see https://google.github.io/googletest/quickstart-cmake.html for info on gtest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# find_package(lager::gncpy_backend CONFIG REQUIRED)

# can add other testing directories here such as integration
add_subdirectory(unit)