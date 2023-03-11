#---------------------------------------------------------------------------
# setup Double Integrator dynamics tests
#---------------------------------------------------------------------------
add_executable(
  double_int_dyn_test
  DoubleIntegrator.cpp
)
target_link_libraries(
  double_int_dyn_test
  GTest::gtest_main
  ${MY_PROJECT}
)
gtest_discover_tests(double_int_dyn_test)