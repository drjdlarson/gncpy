# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:35:30 2020

@author: ryan4
"""
import os.path as path
import pytest
import numpy as np

import gncpy.filters as filters


@pytest.fixture(scope="session")
def Q():
    return 10**-3 * np.eye(2)


@pytest.fixture(scope="session")
def R():
    return np.array([0.1]).reshape((1, 1))


@pytest.fixture(scope="session")
def state_func_list():
    def f1(x, u, **kwargs):
        return 2 * x[0] / (1 - x[0]) + u[0]

    def f2(x, u, **kwargs):
        return (1 - 2 * x[1]) / 5 - u[0]**2

    return [f1, f2]


@pytest.fixture(scope="session")
def inv_state_func_list():
    def f1(x, u, **kwargs):
        return (x[0] - u[0]) / (x[0] + 2 + u[0])

    def f2(x, u, **kwargs):
        return 0.5 * (1 - 5 * x[1] + 5 * u[0]**2)

    return [f1, f2]


@pytest.fixture(scope="function")
def kalmanFilter():
    filt = filters.KalmanFilter()
    filt.set_state_mat(mat=np.array([[1, -1], [0, 1]]))
    filt.set_input_mat(mat=np.array([[1], [0]]))
    filt.set_proc_noise(mat=np.array([[1.00000033333333e-13, -5e-20],
                                      [-5e-20, 1e-19]]))
    filt.cov = np.diag(np.array([0.0001, 1e-12]))
    filt.set_meas_mat(mat=np.array([[1, 0]]))
    filt.meas_noise = np.array([[(17e-6)**2]])

    return filt


@pytest.fixture(scope="function")
def extKalmanFilter():
    filt = filters.ExtendedKalmanFilter()
    filt.cov = np.array([[0.100116534585999, 10.0022808399544],
                         [10.0022808399544, 1000.29294598339]])
    filt.set_meas_mat(mat=np.array([[1, 0]]))
    filt.meas_noise = np.array([[0.01**2]])
    filt.proc_map = np.array([0, 1]).reshape((2, 1))
    filt.proc_cov = np.diag(np.array([0.2]))

    def f0(x, u, **kwargs):
        return x[1]

    def f1(x, u, **kwargs):
        return -2 * 1.5 * (x[0]**2 - 1) * x[1] - 1.2 * x[0]

    filt.dyn_fncs = [f0, f1]

    return filt


@pytest.fixture(scope="function")
def stFilt():
    filt = filters.StudentsTFilter()

    T = 0.5
    Id = np.eye(2)
    Z = np.zeros((2, 2))

    F = np.vstack((np.hstack((Id, T * Id)),
                   np.hstack((Z, Id))))
    H = np.hstack((Id, Z))
    Q = np.vstack((np.hstack((T**3 / 3 * Id, T**2 / 2 * Id)),
                  np.hstack((T**2 / 2 * Id, T * Id))))
    R = 100 * Id.copy()

    filt.set_state_mat(mat=F)
    filt.set_input_mat(mat=np.zeros((4, 1)))
    filt.set_proc_noise(mat=Q)

    filt.set_meas_mat(mat=H)
    filt.meas_noise = R

    filt.dof = 3
    filt.meas_noise_dof = 3
    filt.proc_noise_dof = 3

    filt.scale = 10 * Q

    return filt
