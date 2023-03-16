#---------------------------------------------------------------------------
# setup Measurement tests
#---------------------------------------------------------------------------
add_executable(
  measurement_test
  Measurement.cpp
)
target_link_libraries(
  measurement_test
  GTest::gtest_main
  lager::gncpy_backend
)
gtest_discover_tests(measurement_test)