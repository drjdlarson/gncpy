#include <gtest/gtest.h>
#include "gncpy/math/Matrix.h"


TEST(MatrixTest, MatrixAdd) {
    lager::gncpy::matrix::Matrix m1({{0., 1.}, {2., 0.3}});
    lager::gncpy::matrix::Matrix m2({{1., 1.}, {3., 0.7}});


    SUCCEED();
}