"""Useful math utility functions."""
import numpy as np
from copy import deepcopy


def get_jacobian(x, fnc, f_args=(), step_size=10**-7):
    """Calculates the jacobian of a function.

    Numerically calculates the jacobian using the central difference method.

    Parameters
    ----------
    x : numpy array
        The point to evaluate about.
    fnc : callable
        The function to evaluate, must be of the form `f(x, *f_args)`.
    f_args : tuple, optional
        Additional argumets for `fnc`. The default is ().
    step_size : float, optional
        The step size to use when calculating the jacobian. The default is
        10**-7.

    Returns
    -------
    jac : N x 1 numpy array
        The jacobain of the function
    """
    inv_step2 = 1 / (2 * step_size)
    n_vars = x.size
    J = np.zeros((n_vars, 1))
    for ii in range(n_vars):
        x_r = x.copy()
        x_l = x.copy()
        x_r[ii] += step_size
        x_l[ii] -= step_size
        J[ii] = (fnc(x_r, *f_args) - fnc(x_l, *f_args))
    return J * inv_step2


def get_hessian(x, fnc, f_args=(), step_size=np.finfo(float).eps**(1 / 4)):
    """Calculates the hessian of a function.

    Numerically calculates the hessian using the central difference method.

    Parameters
    ----------
    x : numpy array
        DESCRIPTION.
    fnc : callable
        The function to evaluate, must be of the form `f(x, *f_args)`.
    f_args : tuple, optional
        Additional arguments for the function. The default is ().
    step_size : float, optional
        Step size for differentiation. The default is np.finfo(float).eps**(1 / 4).

    Returns
    -------
    N x N array
        Hessian of the function.

    """
    den = 1 / (4 * step_size**2)
    n_vars = x.size
    H = np.zeros((n_vars, n_vars))
    for ii in range(n_vars):
        delta_i = np.zeros(x.shape)
        delta_i[ii] = step_size

        x_ip = x + delta_i
        x_im = x - delta_i
        for jj in range(n_vars):
            # only get upper triangle since hessian is symmetric
            if jj < ii:
                continue
            delta_j = np.zeros(x.shape)
            delta_j[jj] = step_size

            x_ip_jp = x_ip + delta_j
            x_ip_jm = x_ip - delta_j
            x_im_jm = x_im - delta_j
            x_im_jp = x_im + delta_j

            H[ii, jj] = (fnc(x_ip_jp, *f_args) - fnc(x_ip_jm, *f_args)
                         - fnc(x_im_jp, *f_args) + fnc(x_im_jm, *f_args)) * den

    # fill full H matrix from upper triangle
    for ii in range(n_vars):
        for jj in range(n_vars):
            if jj >= ii:
                break
            H[ii, jj] = H[jj, ii]
    return H


def get_state_jacobian(t, x, fncs, f_args, u=None, **kwargs):
    r"""Calculates the jacobian matrix for the state of a state space model.

    Notes
    -----
    Numerically calculates the jacobian using the central difference method
    for the state of the standard statespace model

    .. math::
        \dot{x} = f(t, x, u)


    Parameters
    ----------
    t : float
        timestep to evaluate at.
    x : N x 1 numpy array
        state to calculate the jocobain about.
    fncs : list
        1 function per state in order. They must have the signature
        `f(t, x, u, *f_args)` if `u` is given or `f(t, x, *f_args)` if `u` is
        not given.
    f_args : tuple
        Additional arguemnts to pass to each function in `fncs`.
    u : Nu x 1 numpy array, optional
        the control signal to calculate the jacobian about. The default is
        None.
    \*\*kwargs : dict, optional
        Additional keyword arguments for :meth:`gncpy.math.get_jacobian`.

    Returns
    -------
    jac : N x N numpy array
        Jaccobian matrix.
    """
    n_states = x.size
    A = np.zeros((n_states, n_states))
    for row in range(0, n_states):
        if u is not None:
            res = get_jacobian(x.copy(),
                               lambda _x, *_f_args: fncs[row](t, _x, u, *_f_args),
                               f_args=f_args, **kwargs)
        else:
            res = get_jacobian(x.copy(),
                               lambda _x, *_f_args: fncs[row](t, _x, *_f_args),
                               f_args=f_args, **kwargs)

        A[[row], :] = res.T
    return A


def get_input_jacobian(t, x, u, fncs, f_args, **kwargs):
    r"""Calculates the jacobian matrix for the input of a state space model.

    Notes
    -----
    Numerically calculates the jacobian using the central difference method
    for the input of the standard statespace model

    .. math::
        \dot{x} = f(t, x, u)


    Parameters
    ----------
    t : float
        timestep to evaluate at.
    x : N x 1 numpy array
        state to calculate the jocobain about.
    u : Nu x 1 numpy array
        control input to calculate the jocobian about.
    fncs : list
        1 function per state in order. They must have the signature
        `f(t, x, u, *f_args)` if `u` is given or `f(t, x, *f_args)` if u is
        not given.
    f_args : tuple
        Additional arguemnts to pass to each function in `fncs`.
    \*\*kwargs : dict, optional
        Additional keyword arguments for :meth:`gncpy.math.get_jacobian`.

    Returns
    -------
    jac : N x Nu numpy array
        jacobian matrix.

    """
    n_states = x.size
    n_inputs = u.size
    B = np.zeros((n_states, n_inputs))
    for row in range(0, n_states):
        res = get_jacobian(u.copy(),
                           lambda _u, *_f_args: fncs[row](t, x, _u, *_f_args),
                           **kwargs)
        B[[row], :] = res.T
    return B


def rk4(f, x, h, **kwargs):
    """Implements a classic Runge-Kutta integration RK4.

    Parameters
    ----------
    f : callable
        function to integrate, must take x as the first argument and arbitrary
        kwargs after
    x : numpy array, or float
        state needed by function
    h : float
        step size

    Returns
    -------
    state : numpy array, or float
        Integrated state
    """
    k1 = h * f(x, **kwargs)
    k2 = h * f(x + 0.5 * k1, **kwargs)
    k3 = h * f(x + 0.5 * k2, **kwargs)
    k4 = h * f(x + k3, **kwargs)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_backward(f, x, h, **kwargs):
    """Implements a backwards classic Runge-Kutta integration RK4.

    Parameters
    ----------
    f : callable
        function to reverse integrate, must take x as the first argument and
        arbitrary kwargs after
    x : numpy array, or float
        state needed by function
    h : float
        step size

    Returns
    -------
    state : numpy array, or float
        Reverse integrated state
    """
    k1 = f(x, **kwargs)
    k2 = f(x - 0.5 * h * k1, **kwargs)
    k3 = f(x - 0.5 * h * k2, **kwargs)
    k4 = f(x - h * k3, **kwargs)
    return x - (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def log_sum_exp(lst):
    """Utility function for a log-sum-exponential trick.

    Parameters
    ----------
    lst : list
        list of values.

    Returns
    -------
    tot : float
        result of log-sum-exponential calculation.
    """
    if len(lst) == 0:
        return None
    m_val = max(lst)
    tot = 0
    for x in lst:
        tot = tot + np.exp(x - m_val)
    tot = np.log(tot) + m_val
    return tot


def gamma_fnc(alpha):
    r"""Implements a gamma function.

    Notes
    -----
    This implements the gamma function as

    .. math::
        \Gamma(\alpha) = (\alpha -1)!

    Todo
    ----
    Add support for complex number input

    Parameters
    ----------
    alpha : int
        number to evaluate the gamma function at.

    Returns
    -------
    int
        result of the gamma function.

    """
    return np.math.factorial(int(alpha - 1))


def get_elem_sym_fnc(z):
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
            tmp = i_n
            i_n = i_nminus
            i_nminus = tmp
        esf = np.hstack((np.array([[1]]), F[[i_nminus - 1], :]))
        esf = esf.reshape((esf.size, 1))
    return esf


def weighted_sum_vec(w_lst, x_lst):
    """Calculates the weighted sum of a list of vectors.

    Parameters
    ----------
    w_lst : list of floats, or N numpy array
        list of weights.
    x_lst : list of n x 1 numpy arrays, or N x n x 1 numpy array
        list of vectors to be weighted and summed.

    Returns
    -------
    w_sum : n x 1 numpy array
        weighted sum of inputs.
    """
    if isinstance(x_lst, list):
        x = np.stack(x_lst)
    else:
        x = x_lst
    if isinstance(w_lst, list):
        w = np.array(w_lst)
    else:
        w = w_lst
    return np.sum(w.reshape((-1,) + (1,) * (x.ndim - 1)) * x, axis=0)


def weighted_sum_mat(w_lst, P_lst):
    """Calculates the weighted sum of a list of matrices.

    Parameters
    ----------
    w_lst : list of floats or numpy array
        list of weights.
    P_lst : list of n x m numpy arrays or N x n x n numpy array
        list of matrices to be weighted and summed.

    Returns
    -------
    w_sum : n x m numpy array
        weighted sum of inputs.
    """
    if isinstance(P_lst, list):
        cov = np.stack(P_lst)
    else:
        cov = P_lst
    if isinstance(w_lst, list):
        w = np.array(w_lst)
    else:
        w = w_lst
    return np.sum(w.reshape((-1,) + (1,)*(cov.ndim - 1)) * cov, axis=0)


def gaussian_kernel(x, sig):
    """Implements a Gaussian Kernel.

    Parameters
    ----------
    x : float
        point to evaluate the kernel at.
    sig : float
        kernel parameter.

    Returns
    -------
    float
        kernel value.

    """
    return np.exp(-x**2 / (2 * sig**2))


def epanechnikov_kernel(x):
    """Implements the Epanechnikov kernel.

    Parameters
    ----------
    x : numpy array
        state to evaluate the kernel at

    Returns
    -------
    val : float
        kernal value
    """
    def calc_vn(n):
        if n == 1:
            return 2
        elif n == 2:
            return np.pi
        elif n == 3:
            return 4 * np.pi / 3
        else:
            return 2 * calc_vn(n - 2) / n

    n = x.size
    mag2 = np.sum(x**2)
    if mag2 < 1:
        vn = calc_vn(n)
        val = (x.size + 2) / (2 * vn) * (1 - mag2)
    else:
        val = 0
    return val
