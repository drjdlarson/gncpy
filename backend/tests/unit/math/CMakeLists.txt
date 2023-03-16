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
  lager::gncpy_backend
)
gtest_discover_tests(matrix_test)


#---------------------------------------------------------------------------
# setup Vector tests
#---------------------------------------------------------------------------
add_executable(
  vector_test
  Vector.cpp
)
target_link_libraries(
  vector_test
  GTest::gtest_main
  lager::gncpy_backend
)
gtest_discover_tests(vector_test)


#---------------------------------------------------------------------------
# setup Math tests
#---------------------------------------------------------------------------
add_executable(
  math_test
  Math.cpp
)
target_link_libraries(
  math_test
  GTest::gtest_main
  lager::gncpy_backend
)
gtest_discover_tests(math_test)