#---------------------------------------------------------------------------
# setup Filter tests
#---------------------------------------------------------------------------
add_executable(
  filter_test
  Filters.cpp
)
target_link_libraries(
  filter_test
  GTest::gtest_main
  lager::gncpy_backend
)
gtest_discover_tests(filter_test)