#include <math.h>
#include <gtest/gtest.h>
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Vector.h"
#include "gncpy/math/Math.h"


TEST(MathTest, JacobianVec) {
    lager::gncpy::matrix::Vector x({3., 2.7, -6.25});
    lager::gncpy::matrix::Vector u({-0.5, 2.});
    lager::gncpy::matrix::Vector exp({0.42737987371310737, -21.462216395207179, 8.0999999951814061});

    auto fnc = [&u](const lager::gncpy::matrix::Vector<double>& x_){ return x_(0) * sin(x_(1)) + 3 * x_(2) * x_(1) + u(0); };
    lager::gncpy::matrix::Vector res = lager::gncpy::math::getJacobian(x, fnc);

    for(uint8_t ii = 0; ii < res.size(); ii++) {
        EXPECT_DOUBLE_EQ(exp(ii), res(ii));
    }

    SUCCEED();
}


TEST(MathTest, JacobianSquareMat) {
    lager::gncpy::matrix::Vector x({3., 2.7, -6.25});
    lager::gncpy::matrix::Vector u({-0.5, 2.});

    auto f0 = [&u](const lager::gncpy::matrix::Vector<double>& x_){ return x_(0) * sin(x_(1)) + 3. * x_(2) * x_(1) + u(0); };
    auto f1 = [&u](const lager::gncpy::matrix::Vector<double>& x_){ return x_(0)*x_(0) + 3. * x_(2) * x_(1) + u(0) * u(1); };
    auto f2 = [&u](const lager::gncpy::matrix::Vector<double>& x_){ return x_(2) * cos(x_(0)) + x_(1)*x_(1) + sin(u(0)); };

    std::vector<std::function<double (const lager::gncpy::matrix::Vector<double>&)>> fncLst({f0, f1, f2});

    lager::gncpy::matrix::Matrix res = lager::gncpy::math::getJacobian(x, fncLst);
    lager::gncpy::matrix::Matrix exp({{0.42737987371310737, -21.462216395207179, 8.0999999951814061},
                                      {5.999999999062311, -18.749999988187938, 8.0999999951814061},
                                      {0.88200003744987043, 5.4000000027087935, -0.98999249686926305}});
    
    uint8_t nRows = res.shape()[0];
    uint8_t nCols = res.shape()[1];

    EXPECT_EQ(exp.numRows(), nRows);
    EXPECT_EQ(exp.numCols(), nCols);

    for(uint8_t r = 0; r < nRows; r++) {
        for(uint8_t c = 0; c < nCols; c++) {
            EXPECT_DOUBLE_EQ(exp(r, c), res(r, c));
        }
    }

    SUCCEED();

}


TEST(MathTest, JacobianMat) {
    lager::gncpy::matrix::Vector x({3., 2.7, -6.25});
    lager::gncpy::matrix::Vector u({-0.5, 2.});

    auto f0 = [&x](const lager::gncpy::matrix::Vector<double>& u_){ return x(0) * sin(x(1)) + 3. * x(2) * x(1) + u_(0); };
    auto f1 = [&x](const lager::gncpy::matrix::Vector<double>& u_){ return x(0)*x(0) + 3. * x(2) * x(1) + u_(0) * u_(1); };
    auto f2 = [&x](const lager::gncpy::matrix::Vector<double>& u_){ return x(2) * cos(x(0)) + x(1)*x(1) + sin(u_(0)); };

    std::vector<std::function<double (const lager::gncpy::matrix::Vector<double>&)>> fncLst({f0, f1, f2});

    lager::gncpy::matrix::Matrix res = lager::gncpy::math::getJacobian(u, fncLst);
    lager::gncpy::matrix::Matrix exp({{1.0, 0.0},
                                      {2.0, -0.5},
                                      {0.877582562, 0.0}});
    
    uint8_t nRows = res.shape()[0];
    uint8_t nCols = res.shape()[1];

    EXPECT_EQ(exp.numRows(), nRows);
    EXPECT_EQ(exp.numCols(), nCols);

    for(uint8_t r = 0; r < nRows; r++) {
        for(uint8_t c = 0; c < nCols; c++) {
            EXPECT_NEAR(exp(r, c), res(r, c), 1e-6);
        }
    }

    SUCCEED();

}


TEST(MathTest, GaussianPDF) {
    lager::gncpy::matrix::Vector<double> x({0.5,});
    lager::gncpy::matrix::Vector<double> m({1,});
    lager::gncpy::matrix::Matrix<double> cov({{2,}});

    double res = lager::gncpy::math::calcGaussianPDF(x, m, cov);
    double exp = 0.26500353;

    EXPECT_NEAR(exp, res, 1e-8);

    SUCCEED();
}


TEST(MathTest, GaussianPDFVec) {
    lager::gncpy::matrix::Vector<double> x({0.5, 3.4});
    lager::gncpy::matrix::Vector<double> m({1, 2});
    lager::gncpy::matrix::Matrix<double> cov({{2, 0}, {0, 2}});

    double res = lager::gncpy::math::calcGaussianPDF(x, m, cov);
    double exp = 0.04579756995735449;

    EXPECT_NEAR(exp, res, 1e-8);

    SUCCEED();
}