#include <gtest/gtest.h>
#include "gncpy/math/Exceptions.h"
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Vector.h"


TEST(MatrixTest, Index) {
    lager::gncpy::matrix::Matrix m({{1, 2}, {5, 1}});

    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(1, 1), 1);
    EXPECT_EQ(m(1, 0), 5);
    EXPECT_THROW(m(2, 0), lager::gncpy::matrix::BadIndex);

    SUCCEED();
}


TEST(MatrixTest, Add) {
    lager::gncpy::matrix::Matrix m1({{0., 1.}, {2., 0.3}});
    lager::gncpy::matrix::Matrix m2({{1., 1.}, {3., 0.7}});

    lager::gncpy::matrix::Matrix m3 = m1 + m2;

    lager::gncpy::matrix::Matrix exp({{1., 2.}, {5., 1.}});

    for(uint8_t r = 0; r < exp.numRows(); r++) {
        for(uint8_t c = 0; c < exp.numCols(); c++) {
            std::cout << m3(r, c);
            EXPECT_DOUBLE_EQ(exp(r, c), m3(r, c));
        }
    }

    SUCCEED();
}


TEST(MatrixTest, Multiply) {
    lager::gncpy::matrix::Matrix m1({{0., 1.}, {2., 3.}});
    lager::gncpy::matrix::Matrix m2({{2., 3.}, {4., 5.}});

    lager::gncpy::matrix::Matrix m3 = m1 * m2;

    lager::gncpy::matrix::Matrix exp({{4., 5.}, {16., 21.}});

    for(uint8_t r = 0; r < exp.numRows(); r++) {
        for(uint8_t c = 0; c < exp.numCols(); c++) {
            std::cout << m3(r, c);
            EXPECT_DOUBLE_EQ(exp(r, c), m3(r, c));
        }
    }

    SUCCEED();
}


TEST(MatrixTest, MultiplyVector) {
    lager::gncpy::matrix::Matrix m1({{1., 2.}, {3., 4.}});
    lager::gncpy::matrix::Vector v2({2., 3.});

    for(uint8_t r = 0; r < v2.numRows(); r++) {
        std::cout << v2(r) << std::endl;
    }

    lager::gncpy::matrix::Vector v3 = m1 * v2;

    lager::gncpy::matrix::Vector exp({8., 18.,});

    for(uint8_t r = 0; r < exp.numRows(); r++) {
        std::cout << v3(r) << std::endl;
        EXPECT_DOUBLE_EQ(exp(r), v3(r));
    }

    SUCCEED();
}


TEST(MatrixTest, TransposeMatrix) {
    lager::gncpy::matrix::Matrix m1({{1., 2., 3., 4.}, {5., 6., 7., 8.}});
    lager::gncpy::matrix::Matrix exp({{1., 5.},{2., 6.,},{3., 7.},{4., 8.}});
    lager::gncpy::matrix::Matrix m2 = m1.transpose();

    for(uint8_t r = 0; r < exp.numRows(); r++) {
        for(uint8_t c = 0; c < exp.numCols(); c++) {
            std::cout << m2(r, c);
            EXPECT_DOUBLE_EQ(exp(r, c), m2(r, c));
        }
    }

    SUCCEED();

}

TEST(MatrixTest, TransposeMatrixInPlace) {
    lager::gncpy::matrix::Matrix m1({{1., 2., 3., 4.}, {5., 6., 7., 8.}});
    lager::gncpy::matrix::Matrix exp({{1., 5.},{2., 6.,},{3., 7.},{4., 8.}});

    m1.transpose(true);

    for(uint8_t r = 0; r < exp.numRows(); r++) {
        for(uint8_t c = 0; c < exp.numCols(); c++) {
            std::cout << m1(r, c);
            EXPECT_DOUBLE_EQ(exp(r, c), m1(r, c));
        }
    }

    SUCCEED();

}