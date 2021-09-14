# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:12:55 2020

@author: ryan4
"""
import pytest
import numpy as np
import numpy.testing as test

import gncpy.math as math


def _f1(t, x, u, *args):
    return x[0] * np.sin(x[1]) + 3 * x[2] * x[1] + u[0]


def _f2(t, x, u, *args):
    return x[0]**2 + 3 * x[2] * x[1] + u[0] * u[1]


def _f3(t, x, u, *args):
    return x[2] * np.cos(x[0]) + x[1]**2 + np.sin(u[0])


func_list = [_f1, _f2, _f3]
x_point = np.array([[3], [2.7], [-6.25]])
u_point = np.array([[-0.5], [2]])


@pytest.mark.incremental
class TestJacobianFunctions:
    def test_get_jacobian(self):
        t = 0
        J = math.get_jacobian(x_point.copy(),
                              lambda x, *f_args: func_list[0](t, x,
                                                              u_point.copy(),
                                                              *f_args))
        J_exp = np.array([0.42737988, -21.46221643, 8.1000000]).reshape((3, 1))
        test.assert_allclose(J, J_exp)

    def test_get_state_jacobian(self):
        t = 0
        A = math.get_state_jacobian(t, x_point.copy(),
                                    func_list, (), u=u_point.copy())
        A_exp = np.vstack((np.array([0.42737988, -21.46221643, 8.1000000]),
                           np.array([6.00000, -18.75000000, 8.1000000]),
                           np.array([0.88200, 5.400000, -0.989992497])))
        test.assert_allclose(A, A_exp)

    def test_get_input_jacobian(self):
        t = 0
        B = math.get_input_jacobian(t, x_point.copy(), u_point.copy(),
                                    func_list, ())
        B_exp = np.vstack((np.array([1, 0]),
                           np.array([2, -0.5]),
                           np.array([0.877582562, 0])))
        test.assert_allclose(B, B_exp)


def test_get_hessian():
    def f(x):
        return x[0]**3 + x[0] * x[1]**2 + x[1] * x[2]
    test_point = np.array([1.0, 3.0, 2.0]).reshape((3, 1))
    exp_H = np.vstack((np.hstack((6. * test_point[0], 2. * test_point[1], 0.)),
                       np.hstack((2. * test_point[1], 2. * test_point[0], 1.)),
                       np.array([0., 1., 0.])))
    H = math.get_hessian(test_point, f)
    test.assert_allclose(H, exp_H)


def test_get_elem_sym_fnc():
    z = np.array([[1.2394e1], [1.5497e-5], [5.3683e-1], [5.2124e-16], [5.6417e-2]])
    s = math.get_elem_sym_fnc(z)
    ans = np.array([1.00000000e+00, 1.29872625e+01, 7.38299885e+00, 3.75369344e-01,
                    5.817091e-06, 0.00000000e+00])
    test.assert_allclose(s.flatten(), ans.flatten(), rtol=1e-5, atol=2e-4)


if __name__ == "__main__":
    testCase = TestJacobianFunctions()

    testCase.test_get_jacobian()
