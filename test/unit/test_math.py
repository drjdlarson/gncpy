# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:12:55 2020

@author: ryan4
"""
import pytest
import numpy as np
import numpy.testing as test

import gncpy.math as math


@pytest.mark.incremental
class TestJacobianFunctions:
    def test_get_jacobian(self, func_list, x_point, u_point):
        J = math.get_jacobian(x_point, lambda x: func_list[0](x, u_point))
        J_exp = np.array([0.42737988, -21.46221643, 8.1000000]).reshape((3, 1))
        test.assert_allclose(J, J_exp)

    def test_get_state_jacobian(self, func_list, x_point, u_point):
        A = math.get_state_jacobian(x_point, u_point, func_list)
        A_exp = np.vstack((np.array([0.42737988, -21.46221643, 8.1000000]),
                           np.array([6.00000, -18.75000000, 8.1000000]),
                           np.array([0.88200, 5.400000, -0.989992497])))
        test.assert_allclose(A, A_exp)

    def test_get_input_jacobian(self, func_list, x_point, u_point):
        B = math.get_input_jacobian(x_point, u_point, func_list)
        B_exp = np.vstack((np.array([1, 0]),
                           np.array([2, -0.5]),
                           np.array([0.877582562, 0])))
        test.assert_allclose(B, B_exp)


def test_get_hessian():
    def f(x):
        return x[0]**3 + x[0] * x[1]**2 + x[1] * x[2]
    test_point = np.array([1.0, 3.0, 2.0]).reshape((3, 1))
    exp_H = np.vstack((np.hstack((6.*test_point[0], 2*test_point[1], 0.)),
                       np.hstack((2.*test_point[1], 2.*test_point[0], 1.)),
                       np.array([0., 1., 0.])))
    H = math.get_hessian(test_point, f)
    test.assert_allclose(H, exp_H)
