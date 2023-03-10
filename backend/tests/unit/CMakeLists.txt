#---------------------------------------------------------------------------
# setup Matrix tests
#---------------------------------------------------------------------------
add_executable(
  matrix_test
  Matrix.cpp
)

target_link_libraries(
  matrix_test
  GTest::gtest_main
  ${MY_PROJECT}
)

gtest_discover_tests(matrix_test)