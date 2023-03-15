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
  ${MY_PROJECT}
)
gtest_discover_tests(measurement_test)