"""Useful math utility functions."""

import warnings

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
        x_r = x.copy().astype(float)
        x_l = x.copy().astype(float)
        x_r[ii] += step_size
        x_l[ii] -= step_size
        J[ii] = fnc(x_r, *f_args) - fnc(x_l, *f_args)
    return J * inv_step2


def get_hessian(x, fnc, f_args=(), step_size=np.finfo(float).eps ** (1 / 4)):
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

        x_ip = x.astype(float) + delta_i
        x_im = x.astype(float) - delta_i
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

            H[ii, jj] = (
                fnc(x_ip_jp, *f_args)
                - fnc(x_ip_jm, *f_args)
                - fnc(x_im_jp, *f_args)
                + fnc(x_im_jm, *f_args)
            ) * den

    # fill full H matrix from upper triangle
    for ii in range(n_vars):
        for jj in range(n_vars):
            if jj >= ii:
                break
            H[ii, jj] = H[jj, ii]
    return H


def linearize_dynamics(state_derivative_fnc, trim_state, trim_input, step_size=1e-7):
    """Linearize a nonlinear dynamics model about a trim point.

    Computes the state and input Jacobians (A and B matrices) using central
    finite differences. The dynamics function should have the form:
    x_dot = f(x, u, t) where x is state, u is input, t is time.

    Parameters
    ----------
    state_derivative_fnc : callable
        Dynamics function with signature f(x, u, t) returning x_dot.
    trim_state : numpy array
        Trim state vector (n_states,).
    trim_input : numpy array
        Trim input vector (n_inputs,).
    step_size : float, optional
        Step size for finite differences. Default is 1e-7.

    Returns
    -------
    A : numpy array
        State Jacobian matrix (n_states, n_states).
    B : numpy array
        Input Jacobian matrix (n_states, n_inputs).

    Examples
    --------
    >>> def f(x, u, t):
    ...     return np.array([x[1], u[0]])  # simple double integrator
    >>> x_trim = np.array([0.0, 0.0])
    >>> u_trim = np.array([0.0])
    >>> A, B = linearize_dynamics(f, x_trim, u_trim)
    """
    trim_state = np.asarray(trim_state).flatten()
    trim_input = np.asarray(trim_input).flatten()
    n_states = len(trim_state)
    n_inputs = len(trim_input)

    # State Jacobian: A = del f/del x
    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        x_plus = trim_state.copy()
        x_minus = trim_state.copy()
        x_plus[i] += step_size
        x_minus[i] -= step_size
        f_plus = state_derivative_fnc(x_plus, trim_input, 0)
        f_minus = state_derivative_fnc(x_minus, trim_input, 0)
        A[:, i] = (f_plus - f_minus) / (2 * step_size)

    # Input Jacobian: B = del f/del u
    B = np.zeros((n_states, n_inputs))
    for i in range(n_inputs):
        u_plus = trim_input.copy()
        u_minus = trim_input.copy()
        u_plus[i] += step_size
        u_minus[i] -= step_size
        f_plus = state_derivative_fnc(trim_state, u_plus, 0)
        f_minus = state_derivative_fnc(trim_state, u_minus, 0)
        B[:, i] = (f_plus - f_minus) / (2 * step_size)

    return A, B


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
            res = get_jacobian(
                x.copy(),
                lambda _x, *_f_args: fncs[row](t, _x, u, *_f_args),
                f_args=f_args,
                **kwargs,
            )
        else:
            res = get_jacobian(
                x.copy(),
                lambda _x, *_f_args: fncs[row](t, _x, *_f_args),
                f_args=f_args,
                **kwargs,
            )

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
        res = get_jacobian(
            u.copy(), lambda _u, *_f_args: fncs[row](t, x, _u, *_f_args), **kwargs
        )
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
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "error",
                            message=".*overflow encountered in double_scalars.*",
                        )
                        try:
                            F[i_n - 1, k - 1] = (
                                z_loc[n - 1] * F[i_nminus - 1, k - 1 - 1]
                            )
                        except RuntimeWarning:
                            F[i_n - 1, k - 1] = np.finfo(float).max
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "error",
                            message=".*overflow encountered in double_scalars.*",
                        )
                        try:
                            F[i_n - 1, k - 1] = (
                                F[i_nminus - 1, k - 1]
                                + z_loc[n - 1] * F[i_nminus - 1, k - 1 - 1]
                            )

                        except RuntimeWarning:
                            F[i_n - 1, k - 1] = np.finfo(float).max
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
    return np.sum(w.reshape((-1,) + (1,) * (cov.ndim - 1)) * cov, axis=0)


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
    return np.exp(-(x**2) / (2 * sig**2))


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


def quat_normalize(q):
    """Normalize a quaternion to unit length.

    Parameters
    ----------
    q : numpy array
        Quaternion [qw, qx, qy, qz] (scalar first)

    Returns
    -------
    numpy array
        Normalized quaternion
    """
    mag = np.linalg.norm(q)
    if mag < np.finfo(float).eps:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / mag


def quat_multiply(q1, q2):
    """Multiply two quaternions using Hamilton convention.

    Computes q1 * q2 in scalar-first format [qw, qx, qy, qz].

    Parameters
    ----------
    q1 : numpy array
        First quaternion [qw, qx, qy, qz]
    q2 : numpy array
        Second quaternion [qw, qx, qy, qz]

    Returns
    -------
    numpy array
        Product quaternion q1 * q2

    Notes
    -----
    For passive rotation quaternions (body-to-NED convention):

    The product q1 * q2 applies q2 first, then q1:
        quat_rotate_vector(quat_multiply(q1, q2), v) = quat_rotate_vector(q1, quat_rotate_vector(q2, v))

    When converting to DCM (NED-to-body), the order reverses:
        quat_to_dcm(quat_multiply(q1, q2)) about quat_to_dcm(q2) @ quat_to_dcm(q1)
        (Because DCM is the transpose of the quaternion rotation)
    """
    qw1, qx1, qy1, qz1 = q1
    qw2, qx2, qy2, qz2 = q2

    return np.array(
        [
            qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2,
            qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2,
            qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2,
            qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2,
        ]
    )


def quat_conjugate(q):
    """Compute the conjugate of a quaternion.

    Parameters
    ----------
    q : numpy array
        Quaternion [qw, qx, qy, qz]

    Returns
    -------
    numpy array
        Conjugate quaternion [qw, -qx, -qy, -qz]

    Notes
    -----
    For unit quaternions (passive rotation quaternions), the conjugate
    reverses the rotation direction:

    - If q represents body-to-NED, then quat_conjugate(q) represents NED-to-body
    - For rotation: quat_rotate_vector(quat_conjugate(q), v) applies inverse rotation
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q):
    """Compute the inverse of a quaternion.

    For unit quaternions, this is equivalent to the conjugate.

    Parameters
    ----------
    q : numpy array
        Quaternion [qw, qx, qy, qz]

    Returns
    -------
    numpy array
        Inverse quaternion
    """
    mag_sq = np.sum(q**2)
    if mag_sq < np.finfo(float).eps:
        raise ValueError("Cannot compute inverse of zero quaternion")
    return quat_conjugate(q) / mag_sq


def quat_rotate_vector(q, v):
    """Rotate a 3D vector by a quaternion using Hamilton convention.

    Computes v' = q * v * q^(-1) where v is treated as a pure quaternion.
    This applies the rotation that q represents.

    Parameters
    ----------
    q : numpy array
        Quaternion [qw, qx, qy, qz] in scalar-first Hamilton convention.
        **If q is a passive rotation quaternion from euler_to_quat, it
        represents body-to-NED rotation.**
    v : numpy array
        3D vector to rotate (must be in the same frame as q's input frame)

    Returns
    -------
    numpy array
        Rotated 3D vector in q's output frame.

    Notes
    -----
    **USAGE WITH PASSIVE ROTATION QUATERNION (BODY-TO-NED):**

    If q is from euler_to_quat (passive rotation quaternion):

    - Body to NED: v_ned = quat_rotate_vector(q, v_body)
      Direct use rotates FROM body TO NED frame

    - NED to body: v_body = quat_rotate_vector(quat_conjugate(q), v_ned)
      Use conjugate to reverse the rotation direction

    - NED to body (alternative): v_body = quat_to_dcm(q) @ v_ned
      The DCM from quat_to_dcm directly gives NED-to-body transformation

    Example:
        q = euler_to_quat(roll, pitch, yaw)  # passive rotation (body-to-NED)
        v_body = np.array([1, 0, 0])  # forward in body frame
        v_ned = quat_rotate_vector(q, v_body)  # rotates to NED frame
    """
    # Convert vector to pure quaternion [0, vx, vy, vz]
    v_quat = np.array([0.0, v[0], v[1], v[2]])

    # Perform rotation: q * v * q_conj
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)

    return result[1:4]


def quat_to_dcm(q):
    """Convert quaternion to direction cosine matrix (DCM).

    Implements equation (6.79) from the textbook. Takes a "passive rotation
    quaternion" (body-to-NED) and produces a NED-to-body DCM.

    Parameters
    ----------
    q : numpy array
        Quaternion [qw, qx, qy, qz] in scalar-first Hamilton convention.
        **Must be a passive rotation quaternion (body-to-NED) from euler_to_quat.**

    Returns
    -------
    numpy array
        3x3 **NED-to-body DCM**. Transforms vectors FROM NED TO body frame:
        v_body = DCM @ v_ned

    Notes
    -----
    **CRITICAL: Quaternion vs DCM Frame Convention**

    This function produces the TRANSPOSE/INVERSE of what the quaternion represents:

    - Input q: passive rotation quaternion (body-to-NED)
      quat_rotate_vector(q, v_body) rotates TO NED frame

    - Output DCM: NED-to-body transformation matrix
      DCM @ v_ned transforms TO body frame

    This is intentional per textbook equation 6.79. The DCM is the transpose
    of the rotation matrix that q represents via q*v*q_conj.

    To get a body-to-NED DCM instead: use DCM.T
    """
    qw, qx, qy, qz = q

    # DCM from quaternion (equation 6.79)
    dcm = np.array(
        [
            [
                qw**2 + qx**2 - qy**2 - qz**2,
                2 * (qx * qy + qw * qz),
                2 * (qx * qz - qw * qy),
            ],
            [
                2 * (qx * qy - qw * qz),
                qw**2 - qx**2 + qy**2 - qz**2,
                2 * (qy * qz + qw * qx),
            ],
            [
                2 * (qx * qz + qw * qy),
                2 * (qy * qz - qw * qx),
                qw**2 - qx**2 - qy**2 + qz**2,
            ],
        ]
    )
    return dcm


def dcm_to_quat(dcm):
    """Convert direction cosine matrix (DCM) to quaternion.

    Uses Shepperd's method (equations 6.80-6.83) for numerical stability.
    This function inverts quat_to_dcm by taking a NED-to-body DCM and
    producing a body-to-NED passive rotation quaternion.

    Parameters
    ----------
    dcm : numpy array
        3x3 **NED-to-body DCM** (transforms vectors FROM NED TO body frame):
        v_body = dcm @ v_ned

    Returns
    -------
    numpy array
        **Passive rotation quaternion** [qw, qx, qy, qz] representing
        body-to-NED rotation. Matches the convention of euler_to_quat.

    Notes
    -----
    **INVERTING THE FRAME CONVENTION:**

    This function properly inverts quat_to_dcm:
    - Input: NED-to-body DCM
    - Output: body-to-NED passive rotation quaternion

    The raw Shepperd's equations (6.80-6.83) extract a quaternion matching
    the DCM's frame convention (NED-to-body), so this function returns the
    conjugate to produce the body-to-NED passive rotation quaternion that
    matches euler_to_quat convention.

    Round-trip guarantee: dcm_to_quat(quat_to_dcm(q)) about q
    """
    trace = np.trace(dcm)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (dcm[2, 1] - dcm[1, 2]) * s
        qy = (dcm[0, 2] - dcm[2, 0]) * s
        qz = (dcm[1, 0] - dcm[0, 1]) * s
    elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[1, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
        qw = (dcm[2, 1] - dcm[1, 2]) / s
        qx = 0.25 * s
        qy = (dcm[0, 1] + dcm[1, 0]) / s
        qz = (dcm[0, 2] + dcm[2, 0]) / s
    elif dcm[1, 1] > dcm[2, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
        qw = (dcm[0, 2] - dcm[2, 0]) / s
        qx = (dcm[0, 1] + dcm[1, 0]) / s
        qy = 0.25 * s
        qz = (dcm[1, 2] + dcm[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
        qw = (dcm[1, 0] - dcm[0, 1]) / s
        qx = (dcm[0, 2] + dcm[2, 0]) / s
        qy = (dcm[1, 2] + dcm[2, 1]) / s
        qz = 0.25 * s

    # Return conjugate to match euler_to_quat convention (body-to-NED)
    # The raw calculation gives NED-to-body quaternion, but we want body-to-NED
    return np.array([qw, -qx, -qy, -qz])


def quat_to_euler(q):
    """Convert quaternion to Euler angles (3-2-1 sequence).

    Implements equation (6.84) from the textbook. Extracts Euler angles
    from a "passive rotation quaternion" (body-to-NED).

    Parameters
    ----------
    q : numpy array
        Quaternion [qw, qx, qy, qz] representing body-to-NED rotation.
        **Must be a passive rotation quaternion matching euler_to_quat convention.**

    Returns
    -------
    roll : float
        Roll angle in radians (rotation about x-axis, θx)
    pitch : float
        Pitch angle in radians (rotation about y-axis, θy)
    yaw : float
        Yaw angle in radians (rotation about z-axis, θz)

    Notes
    -----
    Round-trip guarantee: quat_to_euler(euler_to_quat(r, p, y)) about (r, p, y)

    This function expects a passive rotation quaternion (body-to-NED) as
    produced by euler_to_quat or dcm_to_quat.
    """
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles (3-2-1 sequence) to quaternion.

    Implements equation (6.85) from the textbook. Produces a "passive rotation
    quaternion" (body-to-NED) as described in the textbook.

    Parameters
    ----------
    roll : float
        Roll angle in radians (rotation about x-axis, θx in textbook)
    pitch : float
        Pitch angle in radians (rotation about y-axis, θy in textbook)
    yaw : float
        Yaw angle in radians (rotation about z-axis, θz in textbook)

    Returns
    -------
    numpy array
        Quaternion [qw, qx, qy, qz] in scalar-first Hamilton convention.
        **Passive rotation quaternion: represents body-to-NED transformation.**

    Notes
    -----
    **PASSIVE ROTATION QUATERNION CONVENTION (BODY-TO-NED):**

    The quaternion q returned by this function is a "passive rotation quaternion"
    (textbook terminology) representing body-to-NED transformation:

    - Direct rotation: quat_rotate_vector(q, v_body)  to  v_ned
      Rotates body-frame vectors TO NED frame using q*v*q_conj

    - DCM conversion: quat_to_dcm(q)  to  NED-to-body DCM
      Produces the TRANSPOSE/INVERSE of the rotation (NED to body)
      This is intentional per textbook equation 6.79

    - Inverse rotation: quat_rotate_vector(quat_conjugate(q), v_ned)  to  v_body
      Use q_conj for NED to body vector transformations

    Summary: Quaternion is body-to-NED, but quat_to_dcm gives NED-to-body DCM.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


def quat_slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions.

    Parameters
    ----------
    q1 : numpy array
        Start quaternion [qw, qx, qy, qz]
    q2 : numpy array
        End quaternion [qw, qx, qy, qz]
    t : float
        Interpolation parameter in [0, 1]

    Returns
    -------
    numpy array
        Interpolated quaternion
    """
    # Ensure unit quaternions
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    # Compute dot product
    dot = np.dot(q1, q2)

    # If dot product is negative, negate one quaternion to take shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return quat_normalize(result)

    # Perform slerp
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)

    w1 = np.sin((1.0 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2
