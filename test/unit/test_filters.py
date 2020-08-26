# -*- coding: utf-8 -*-
import pytest
import numpy as np
import numpy.testing as test


@pytest.mark.incremental
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
