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

TEST(MatrixTest, MultiplyNonSquare) {
    lager::gncpy::matrix::Matrix<double> m1({{0, 1}, {2, 3}});
    lager::gncpy::matrix::Matrix<double> m2({{1, 2, 3}, {-1, -2, -3}});
    lager::gncpy::matrix::Matrix<double> m3=m1*m2;

    lager::gncpy::matrix::Matrix<double> exp({{-1, -2, -3}, {-1, -2, -3}});

    for (uint8_t r=0;r<exp.numRows();r++) {
        for (uint8_t c=0;c<exp.numCols();c++) {
            EXPECT_EQ(exp(r,c), m3(r,c));
        }
    }
    SUCCEED();
}


TEST(MatrixTest, MultiplyVector) {
    lager::gncpy::matrix::Matrix m1({{1., 2.}, {3., 4.}});
    lager::gncpy::matrix::Matrix m2({{1., 0., 0.}, {0., 1., 0.}});
    lager::gncpy::matrix::Vector v1({2., 3.});
    lager::gncpy::matrix::Vector v2({4., 5., 6.});

    std::cout << "v1 = \n" << v1 << std::endl;
    std::cout << "v2 = \n" << v1 << std::endl;
    std::cout << "m1 = \n" << m1 << std::endl;
    std::cout << "m2 = \n" << m2 << std::endl;

    lager::gncpy::matrix::Vector res1 = m1 * v1;
    lager::gncpy::matrix::Vector res2 = m2 * v2;
    lager::gncpy::matrix::Vector exp1({8., 18.,});
    lager::gncpy::matrix::Vector exp2({v2(0), v2(1),});

    EXPECT_EQ(exp1.size(), res1.size());
    for(uint8_t r = 0; r < exp1.numRows(); r++) {
        std::cout << res1(r) << std::endl;
        EXPECT_DOUBLE_EQ(exp1(r), res1(r));
    }

    EXPECT_EQ(exp2.size(), res2.size());
    for(uint8_t r = 0; r < exp2.numRows(); r++) {
        std::cout << res2(r) << std::endl;
        EXPECT_DOUBLE_EQ(exp2(r), res2(r));
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

TEST(MatrixTest, InovCovCalc) {
    lager::gncpy::matrix::Matrix<double> m1({{1, 0, 0, 0}, {0, 1, 0, 0}});
    lager::gncpy::matrix::Matrix<double> m2({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
    // lager::gncpy::matrix::Matrix<double> exp({{1, 0}, {0, 1}})
    lager::gncpy::matrix::Matrix<double> exp({{1, 0}, {0, 1}, {0, 0}, {0, 0}});

    auto out = lager::gncpy::matrix::Matrix<double>({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}) * lager::gncpy::matrix::Matrix<double>({{1, 0}, {0, 1}, {0, 0}, {0, 0}});

    std::cout<<out<<"\n"<<lager::gncpy::matrix::Matrix<double>({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}) * lager::gncpy::matrix::Matrix<double>({{1, 0}, {0, 1}, {0, 0}, {0, 0}});

    for (uint8_t r=0;r<exp.numRows();r++) {
        for (uint8_t c=0;c<exp.numCols();c++) {
            EXPECT_EQ(out(r, c), exp(r, c));
        }
    }

    // std::cout<<m2*m1.transpose()<<"\n"<<m1.transpose();
    SUCCEED();
}