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
  ${MY_PROJECT}
)
gtest_discover_tests(filter_test)