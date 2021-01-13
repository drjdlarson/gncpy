# -*- coding: utf-8 -*-
"""
This file contains useful math utility functions.

"""
import numpy as np
import scipy.linalg as la
from copy import deepcopy


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
        J[ii] = (fnc(x_r, **kwargs) - fnc(x_l, **kwargs)) * inv_step2
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
                                   lambda x_: fncs[row](x_, u, **kwargs)).T
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


def rk4(f, x, h, **kwargs):
    """ Implements a classic Runge-Kutta integration RK4.

    Args:
        f (function): function to integrate, must take x as the first argument
            and arbitrary kwargs after
        x (numpy array, or float): state needed by function
        h (float): step size
    Returns:
        (numpy array, or float): Integrated state
    """
    k1 = h * f(x, **kwargs)
    k2 = h * f(x + 0.5 * k1, **kwargs)
    k3 = h * f(x + 0.5 * k2, **kwargs)
    k4 = h * f(x + k3, **kwargs)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_backward(f, x, h, **kwargs):
    """ Implements a backwards classic Runge-Kutta integration RK4.

    Args:
        f (function): function to reverse integrate, must take x as the first
            argument and arbitrary kwargs after
        x (numpy array, or float): state needed by function
        h (float): step size
    Returns:
        (numpy array, or float): Reverse integrated state
    """
    k1 = f(x, **kwargs)
    k2 = f(x - 0.5 * h * k1, **kwargs)
    k3 = f(x - 0.5 * h * k2, **kwargs)
    k4 = f(x - h * k3, **kwargs)
    return x - (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def log_sum_exp(lst):
    if len(lst) == 0:
        return None
    m_val = max(lst)
    tot = 0
    for x in lst:
        tot = tot + np.exp(x - m_val)
    tot = np.log(tot) + m_val
    return tot


def disrw(F, G, dt, Rwpsd):
    ME = np.vstack((np.hstack((-F, G @ Rwpsd @ G.T)),
                    np.hstack((np.zeros(F.shape), F.T))))
    phi = la.expm(ME * dt)
    phi_12 = phi[0:F.shape[0], F.shape[1]:]
    phi_22 = phi[F.shape[0]:, F.shape[1]:]

    return phi_22.T @ phi_12


def gamma_fnc(alpha):
    return np.math.factorial(alpha - 1)


def get_elem_sym_fnc(z):
    """
    Words
    """
    if z.size == 0:
        esf = np.array([[1]])
    else:
        z_loc = deepcopy(z).reshape(z.size)
        i_n = 1
        i_nminus = 2

        n_z = z_loc.size
        F = np.zeros((2, n_z))

        for n in range(1, n_z + 1):
            F[i_n - 1, 0] = F[i_nminus - 1, 0] + z_loc[n - 1]
            for k in range(2, n + 1):
                if k == n:
                    F[i_n - 1, k - 1] = z_loc[n - 1] * F[i_nminus - 1, k - 1 - 1]
                else:
                    F[i_n - 1, k - 1] = F[i_nminus - 1, k - 1] \
                        + z_loc[n - 1] * F[i_nminus - 1, k - 1 - 1]
            tmp = i_n;
            i_n = i_nminus
            i_nminus = tmp
        esf = np.hstack((np.array([[1]]), F[[i_nminus - 1], :]))
        esf = esf.reshape((esf.size, 1))
    return esf


def weighted_sum_vec(w_lst, x_lst):
    """ Calculates the weighted sum of a list of vectors

    Args:
        w_lst (list of floats): list of weights.
        x_lst (list of n x 1 numpy arrays): list of vectors to be weighted and
            summed.

    Returns:
        n x 1 numpy array: weighted sum of inputs.
    """
    return np.sum([w * x for w, x in zip(w_lst, x_lst)], axis=0)


def weighted_sum_mat(w_lst, P_lst):
    """ Calculates the weighted sum of a list of matrices

    Args:
        w_lst (list of floats): list of weights.
        P_lst (list of n x m numpy arrays): list of matrices to be weighted and
            summed.

    Returns:
        n x m numpy array: weighted sum of inputs.
    """
    return np.sum([w * P for w, P in zip(w_lst, P_lst)], axis=0)
