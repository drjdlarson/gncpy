#include <gtest/gtest.h>
#include "gncpy/math/Vector.h"


TEST(VectorTest, Index) {
    lager::gncpy::matrix::Vector v({2, 3});

    EXPECT_EQ(v(0), 2);
    EXPECT_EQ(v(1), 3);

    SUCCEED();
}

TEST(VectorTest, Add) {
    lager::gncpy::matrix::Vector v1({2, 3});
    lager::gncpy::matrix::Vector v2({4, 5});

    lager::gncpy::matrix::Vector<int> res = v1 + v2;
    lager::gncpy::matrix::Vector exp({6, 8});

    EXPECT_EQ(exp(0), res(0));
    EXPECT_EQ(exp(1), res(1));

    SUCCEED();
}

TEST(VectorTest, Subtract) {
    lager::gncpy::matrix::Vector v1({2, 3});
    lager::gncpy::matrix::Vector v2({1, 1});
    
    lager::gncpy::matrix::Vector<int> res = v1 - v2;
    lager::gncpy::matrix::Vector exp({1, 2});

    EXPECT_EQ(exp(0), res(0));
    EXPECT_EQ(exp(1), res(1));

    SUCCEED();
}