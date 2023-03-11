#include <gtest/gtest.h>
#include "gncpy/math/Vector.h"


TEST(VectorTest, Index) {
    lager::gncpy::matrix::Vector v({2, 3});

    EXPECT_EQ(v(0), 2);
    EXPECT_EQ(v(1), 3);

    SUCCEED();
}
