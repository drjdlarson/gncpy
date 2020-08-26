# -*- coding: utf-8 -*-
import pytest
import numpy as np
import numpy.testing as test


class TestKalmanFilter:
    def test_predict(self, kalmanFilter):
        x0 = np.array([-1.10332340082511e-05, 0]).reshape((2, 1))
        u = np.array([[0.00110032804186152]])

        x = kalmanFilter.predict(cur_state=x0, cur_input=u)

        exp_x = np.array([0.00108929480785327, 0]).reshape((2, 1))
        exp_cov = np.array([[0.0001000000011, -1.00000005e-12],
                            [-1.00000005e-12, 1.0000001e-12]])

        test.assert_allclose(x, exp_x)
        test.assert_allclose(kalmanFilter.cov, exp_cov)

    def test_correct(self, kalmanFilter):
        kalmanFilter.cov = np.array([[2.90099164810315e-10, -1.00000005e-12],
                                     [-1.00000005e-12, 1.0000001e-12]])
        x0 = np.array([0.00108929480785327, 0]).reshape((2, 1))
        y = np.array([[0.00112007982271341]])

        x = kalmanFilter.correct(cur_state=x0, meas=y)

        exp_x = np.array([0.00110471653118199,
                          -5.31601809674181e-08]).reshape((2, 1))
        exp_cov = np.array([[1.4477426963246e-10, -4.99050995082444e-13],
                            [-4.99050995082444e-13, 9.98273279861471e-13]])

        test.assert_allclose(x, exp_x)
        test.assert_allclose(kalmanFilter.cov, exp_cov)


class TestExtendedKalmanFilter:
    def test_predict(self, extKalmanFilter):
        extKalmanFilter.cov = np.array([[9.9900216066734e-05,
                                         0.00998066923913823],
                                        [0.00998066923913932,
                                         1.99837897791633]])
        dt = 0.01
        x0 = np.array([1.01659343097534, 1.99448601065866]).reshape((2, 1))
        u = np.array([[0]])

        x = extKalmanFilter.predict(cur_state=x0, cur_input=u, dt=dt)

        exp_x = np.array([1.03646286974202, 1.97896101097449]).reshape((2, 1))
        exp_cov = np.array([[0.000498717421278935, 0.0298766088588289],
                            [0.02987660885883, 1.99104285839628]])

        test.assert_allclose(x, exp_x)
        test.assert_allclose(extKalmanFilter.cov, exp_cov)

    def test_correct(self, extKalmanFilter):
        x0 = np.array([0.996510171108603, -0.0119598396649277]).reshape((2, 1))
        y = np.array([[1.0166134908585]])

        x = extKalmanFilter.correct(cur_state=x0, meas=y)

        exp_x = np.array([1.01659343097534, 1.99448601065866]).reshape((2, 1))
        exp_cov = np.array([[9.9900216066734e-05, 0.00998066923913823],
                            [0.00998066923913932, 1.99837897791633]])

        test.assert_allclose(x, exp_x)
        test.assert_allclose(extKalmanFilter.cov, exp_cov)
