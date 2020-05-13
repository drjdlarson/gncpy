# -*- coding: utf-8 -*-
"""
This file contains useful math utility functions.

"""
import numpy as np


def get_jacobian(x, fnc, **kwargs):
    """Calculates the jacobian of a function.

    Numerically calculates the jacobian using the central difference method.

    Args:
        x (numpy array): The point to evaluate at
        fnc (function): The function to evaluate

    Returns:
        (Nx1 numpy array): The jacobain of the function
    """
    step_size = kwargs.get('step_size', 10**-7)
    inv_step2 = 1 / (2 * step_size)
    n_vars = x.size
    J = np.zeros((n_vars, 1))
    for ii in range(0, n_vars):
        x_r = x.copy()
        x_l = x.copy()
        x_r[ii] += step_size
        x_l[ii] -= step_size
        J[ii] = (fnc(x_r) - fnc(x_l)) * inv_step2
    return J


def get_hessian(x, fnc, **kwargs):
    """Calculates the hessian of a function.

    Numerically calculates the hessian using the central difference method.

    Args:
        x (numpy array): The point to evaluate at
        fnc (function): The function to evaluate

    Returns:
        (NxN numpy array): The hessian of the function
    """
    step_size = np.finfo(float).eps**(1/4)
    den = 1 / (4 * step_size**2)
    n_vars = x.size
    H = np.zeros((n_vars, n_vars))
    for ii in range(0, n_vars):
        for jj in range(0, n_vars):
            x_ip_jp = x.copy()
            x_ip_jp[ii] += step_size
            x_ip_jp[jj] += step_size

            x_ip_jm = x.copy()
            x_ip_jm[ii] += step_size
            x_ip_jm[jj] -= step_size

            x_im_jm = x.copy()
            x_im_jm[ii] -= step_size
            x_im_jm[jj] -= step_size

            x_im_jp = x.copy()
            x_im_jp[ii] -= step_size
            x_im_jp[jj] += step_size

            H[ii, jj] = (fnc(x_ip_jp) - fnc(x_ip_jm) - fnc(x_im_jp)
                         + fnc(x_im_jm)) * den
    return 0.5 * (H + H.T)


def get_state_jacobian(x, u, fncs, **kwargs):
    r"""Calculates the jacobian matrix for the state of a state space model.

    Numerically calculates the jacobian using the central difference method
    for the state of the standard statespace model

    .. math::
        \dot{x} = Ax + Bu

    Args:
        x (numpy array): The state to evaluate at
        u (numpy array): The input to evaluate at
        fnc (function): The function to evaluate. Must take in x, u

    Returns:
        (Nx1 numpy array): The jacobain of the function
    """
    n_states = x.size
    A = np.zeros((n_states, n_states))
    for row in range(0, n_states):
        A[[row], :] = get_jacobian(x.copy(),
                                   lambda x_: fncs[row](x_, u, **kwargs),
                                   **kwargs).T
    return A


def get_input_jacobian(x, u, fncs, **kwargs):
    r"""Calculates the jacobian matrix for the input of a state space model.

    Numerically calculates the jacobian using the central difference method
    for the input of the standard statespace model

    .. math::
        \dot{x} = Ax + Bu

    Args:
        x (numpy array): The state to evaluate at
        u (numpy array): The input to evaluate at
        fnc (function): The function to evaluate. Must take in x, u

    Returns:
        (Nx1 numpy array): The jacobain of the function
    """
    n_states = x.size
    n_inputs = u.size
    B = np.zeros((n_states, n_inputs))
    for row in range(0, n_states):
        B[[row], :] = get_jacobian(u.copy(),
                                   lambda u_: fncs[row](x, u_, **kwargs),
                                   **kwargs).T
    return B
