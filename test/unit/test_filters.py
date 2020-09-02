# -*- coding: utf-8 -*-
import pytest
import numpy as np
import numpy.testing as test
from scipy.linalg import block_diag

import gncpy.filters as filters


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

        (x, _) = kalmanFilter.correct(cur_state=x0, meas=y)

        exp_x = np.array([0.00110471653118199,
                          -5.31601809674181e-08]).reshape((2, 1))
        exp_cov = np.array([[1.4477426963246e-10, -4.99050995082444e-13],
                            [-4.99050995082444e-13, 9.98273279861471e-13]])

        test.assert_allclose(x, exp_x)
        test.assert_allclose(kalmanFilter.cov, exp_cov)


class TestExtendedKalmanFilter:
    def test_predict1(self, extKalmanFilter):
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

    def test_predict2(self):
        dt = 1.0
        filt = filters.ExtendedKalmanFilter()

        def meas_fnc(state, **kwargs):
            mag = state[0, 0]**2 + state[2, 0]**2
            sqrt_mag = np.sqrt(mag)
            mat = np.vstack((np.hstack((state[2, 0] / mag, 0,
                                        -state[0, 0] / mag, 0, 0)),
                            np.hstack((state[0, 0] / sqrt_mag, 0,
                                       state[2, 0] / sqrt_mag, 0, 0))))
            return mat

        def meas_mod(state, **kwargs):
            z1 = np.arctan2(state[0, 0], state[2, 0])
            z2 = np.sqrt(np.sum(state.flatten()**2))
            return np.array([[z1], [z2]])

        filt.set_meas_mat(fnc=meas_fnc)
        filt.set_meas_model(meas_mod)
        filt.meas_noise = np.diag([(2 * np.pi / 180)**2, 10**2])
        sig_w = 5
        sig_u = np.pi / 180
        G = np.array([[dt**2 / 2, 0, 0],
                      [dt, 0, 0],
                      [0, dt**2 / 2, 0],
                      [0, dt, 0],
                      [0, 0, 1]])
        Q = block_diag(sig_w**2 * np.eye(2), np.array([[sig_u**2]]))
        filt.set_proc_noise(mat=G @ Q @ G.T)

        # returns x_dot
        def f0(x, u, **kwargs):
            return x[1]

        # returns x_dot_dot
        def f1(x, u, **kwargs):
            return -x[4] * x[3]

        # returns y_dot
        def f2(x, u, **kwargs):
            return x[3]

        # returns y_dot_dot
        def f3(x, u, **kwargs):
            return x[4] * x[1]

        # returns omega_dot
        def f4(x, u, **kwargs):
            return 0

        filt.dyn_fncs = [f0, f1, f2, f3, f4]
        x0 = np.array([-1500.34728832552, 0, 1727.74139038008,
                       0, 0]).reshape((5, 1))
        u = np.array([[0]])

        filt.cov = np.array([[129.357317971582, 0, 199.220830906414, 0, 0],
                             [0, 2500, 0, 0, 0],
                             [199.220830906414, 0, 1291.47883159233, 0, 0],
                             [0, 0, 0, 2500, 0],
                             [0, 0, 0, 0, 0.0109662271123215]])
        x = filt.predict(cur_state=x0, cur_input=u, dt=dt)

        exp_x = x0
        exp_cov = np.array([[2635.60731797158, 2512.50000000000,
                             199.220830906414, 0, 0],
                            [2512.50000000000, 2525, 0, 0, 0],
                            [199.220830906414, 0, 3797.72883159233,
                             2512.50000000000, 0],
                            [0, 0, 2512.50000000000, 2525, 0],
                            [0, 0, 0, 0, 0.0112708445321082]])

        test.assert_allclose(x, exp_x)
        test.assert_allclose(filt.cov, exp_cov)

    def test_correct1(self, extKalmanFilter):
        x0 = np.array([0.996510171108603, -0.0119598396649277]).reshape((2, 1))
        y = np.array([[1.0166134908585]])

        (x, _) = extKalmanFilter.correct(cur_state=x0, meas=y)

        exp_x = np.array([1.01659343097534, 1.99448601065866]).reshape((2, 1))
        exp_cov = np.array([[9.9900216066734e-05, 0.00998066923913823],
                            [0.00998066923913932, 1.99837897791633]])

        test.assert_allclose(x, exp_x)
        test.assert_allclose(extKalmanFilter.cov, exp_cov)
        
    def test_correct2(self):
        dt = 1.0
        filt = filters.ExtendedKalmanFilter()

        def meas_fnc(state, **kwargs):
            mag = state[0, 0]**2 + state[2, 0]**2
            sqrt_mag = np.sqrt(mag)
            mat = np.vstack((np.hstack((state[2, 0] / mag, 0,
                                        -state[0, 0] / mag, 0, 0)),
                            np.hstack((state[0, 0] / sqrt_mag, 0,
                                       state[2, 0] / sqrt_mag, 0, 0))))
            return mat

        def meas_mod(state, **kwargs):
            z1 = np.arctan2(state[0, 0], state[2, 0])
            z2 = np.sqrt(np.sum(state.flatten()**2))
            return np.array([[z1], [z2]])

        filt.set_meas_mat(fnc=meas_fnc)
        filt.set_meas_model(meas_mod)
        filt.meas_noise = np.diag([(2 * np.pi / 180)**2, 10**2])
        sig_w = 5
        sig_u = np.pi / 180
        G = np.array([[dt**2 / 2, 0, 0],
                      [dt, 0, 0],
                      [0, dt**2 / 2, 0],
                      [0, dt, 0],
                      [0, 0, 1]])
        Q = block_diag(sig_w**2 * np.eye(2), np.array([[sig_u**2]]))
        filt.set_proc_noise(mat=G @ Q @ G.T)

        # returns x_dot
        def f0(x, u, **kwargs):
            return x[1]

        # returns x_dot_dot
        def f1(x, u, **kwargs):
            return -x[4] * x[3]

        # returns y_dot
        def f2(x, u, **kwargs):
            return x[3]

        # returns y_dot_dot
        def f3(x, u, **kwargs):
            return x[4] * x[1]

        # returns omega_dot
        def f4(x, u, **kwargs):
            return 0

        filt.dyn_fncs = [f0, f1, f2, f3, f4]
        x0 = np.array([-1500.34728832552, 0, 1727.74139038008,
                       0, 0]).reshape((5, 1))
        y = np.array([[0.566508296149022],
                      [1776.65653506119]])

        filt.cov = np.array([[2635.60731797158, 2512.50000000000,
                              199.220830906414, 0, 0],
                             [2512.50000000000, 2525, 0, 0, 0],
                             [199.220830906414, 0, 3797.72883159233,
                              2512.50000000000, 0],
                             [0, 0, 2512.50000000000, 2525, 0],
                             [0, 0, 0, 0, 0.0112708445321082]])

        (x, qz) = filt.correct(cur_state=x0, meas=y)

        exp_x = np.array([-485.172128612191, 959.909976636434,
                          1960.77220788083, 103.813590520219,
                          0]).reshape((5, 1))
        exp_cov = np.array([[1248.90214936017, 1144.44893548648,
                             1012.96290074117, 610.120238438169, 0],
                            [1144.44893548648, 1169.47165607343,
                             919.920270643205, 679.708656792902, 0],
                            [1012.96290074117, 919.920270643206,
                             987.360159613778, 604.960286061584, 0],
                            [610.120238438169, 679.708656792902,
                             604.960286061584, 1227.35451945511, 0],
                            [0, 0, 0, 0, 0.0112708445321082]])
        exp_qz = 3.19757889467627e-227

        test.assert_allclose(x, exp_x)
        test.assert_allclose(filt.cov, exp_cov)
        test.assert_approx_equal(qz, exp_qz)
