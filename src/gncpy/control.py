"""Implements control algorithms.

Some algorithms may also be used for path planning.
"""
import io
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from PIL import Image
from sys import exit

import gncpy.dynamics.basic as gdyn
import gncpy.math as gmath
import gncpy.plotting as gplot


class LQR:
    r"""Implements a Linear Quadratic Regulator (LQR) controller.

    Notes
    -----
    It can be either an infinite horizon where a single controller gain :math:`K`
    is found or a finite horizon where value iteration is preformed during the
    backward pass to calculate a controller gain matrix at every step of the
    time horizon. The finite horizon case is consistent with a Receding Horizon
    formulation of the controller. If non-linear dyanmics are supplied then they
    are linearized at every step of the finite horizon, or once about the initial
    state for infinite horizon. This can also track references by supplying an
    end state. If infinite horizon the the control input is given by

    .. math::

        u = K (r - x) + k

    if the problem is finite horizon the control input (with or without reference
    tracking) is given by

    .. math::

        u = K x + k

    Attributes
    ----------
    dynObj : :class:`gncpy.dynamics.basic.DynamicsBase`
        Dynamics model to generate control parameters for.
    time_horizon : float
        Length of the time horizon for the controller.
    u_nom : Nu x 1 numpy array
        Nominal control input
    ct_go_mats : Nh+1 x N x N numpy array
        Cost-to-go matrices, 1 per step in the time horizon.
    ct_go_vecs : Nh+1 x N x 1 numpy array
        Cost-to-go vectors, 1 per step in the time horizon.
    feedback_gain : (Nh) x Nu x N numpy array
        Feedback gain matrix. If finite horizon there is 1 per timestep.
    feedthrough_gain : (Nh) x Nu x 1
        Feedthrough gain vector. If finite horizon there is 1 per timestep.
    """

    def __init__(self, time_horizon=float("inf")):
        """Initialize an object.

        Parameters
        ----------
        time_horizon : float, optional
            Time horizon for the controller. The default is float("inf").
        """
        super().__init__()

        self._Q = None
        self._R = None
        self._P = None

        self.dynObj = None
        self._dt = None

        self.time_horizon = time_horizon

        self.u_nom = np.array([])
        self.ct_go_mats = np.array([])
        self.ct_go_vecs = np.array([])
        self.feedback_gain = np.array([])
        self.feedthrough_gain = np.array([])

        self._init_state = np.array([])
        self._end_state = np.array([])

    @property
    def dt(self):
        """Timestep."""
        if self.dynObj is not None and isinstance(
            self.dynObj, gdyn.NonlinearDynamicsBase
        ):
            return self.dynObj.dt
        else:
            return self._dt

    @dt.setter
    def dt(self, val):
        if self.dynObj is not None and isinstance(
            self.dynObj, gdyn.NonlinearDynamicsBase
        ):
            self.dynObj.dt = val
        else:
            self._dt = val

    def set_state_model(self, u_nom, dynObj=None, dt=None):
        """Set the state/dynamics model.

        Parameters
        ----------
        u_nom : Nu x 1 numpy array
            Nominal control input.
        dynObj : :class:`gncpy.dynamics.basic.DynamicsBase`, optional
            System dynamics to control. The default is None.
        dt : float, optional
            Timestep to use. Will update the dynamic object if applicable.
            The default is None.
        """
        self.u_nom = u_nom.reshape((-1, 1))
        self.dynObj = dynObj
        if dt is not None:
            self.dt = dt

    def set_cost_model(self, Q, R, P=None):
        r"""Sets the cost model used.

        Notes
        -----
        This implements an LQR controller for the cost function

        .. math::

            J = \int^{t_f}_0 x^T Q x + u^T R u + u^T P x d\tau

        Parameters
        ----------
        Q : N x N numpy array
            State cost matrix.
        R : Nu x Nu numpy array
            Control input cost matrix.
        P : Nu x N, optional
            State and control correlation cost matrix. The default is None which
            gives a zero matrix.
        """
        self._Q = Q
        self._R = R

        if P is None:
            self._P = np.zeros((self._R.shape[0], self._Q.shape[0]))
        else:
            self._P = P

    def _prop_state(
        self,
        tt,
        x_hat,
        u_hat,
        state_args,
        ctrl_args,
        forward,
        inv_state_args,
        inv_ctrl_args,
    ):
        """Propagate the state in time."""
        if self.dynObj is not None:
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                if forward and self.dynObj.dt < 0:
                    self.dynObj.dt *= -1  # set to go forward
                elif not forward and self.dynObj.dt > 0:
                    self.dynObj.dt *= -1

            if forward:
                return self.dynObj.propagate_state(
                    tt, x_hat, u=u_hat, state_args=state_args, ctrl_args=ctrl_args
                )
            else:
                return self.dynObj.propagate_state(
                    tt,
                    x_hat,
                    u=u_hat,
                    state_args=inv_state_args,
                    ctrl_args=inv_ctrl_args,
                )

        else:
            raise NotImplementedError()

    def get_state_space(self, tt, x_hat, u_hat, state_args, ctrl_args):
        """Return the :math:`A, B` matrices.

        Parameters
        ----------
        tt : float
            Current timestep.
        x_hat : N x 1 numpy array
            Current state.
        u_hat : Nu x 1 numpy array
            Current control input.
        state_args : tuple
            Extra arguments for the state calculation.
        ctrl_args : tuple
            Extra arguments for the control calculation.

        Raises
        ------
        NotImplementedError
            If unsupported dynamics are used.

        Returns
        -------
        A : N x N numpy array
            State matrix.
        B : N x Nu numpy array
            Input matrix.
        """
        if self.dynObj is not None:
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                if self.dynObj.dt < 0:
                    self.dynObj.dt *= -1  # flip back to forward to get forward matrices

                A = self.dynObj.get_state_mat(
                    tt, x_hat, *state_args, u=u_hat, ctrl_args=ctrl_args
                )
                B = self.dynObj.get_input_mat(tt, x_hat, u_hat, *ctrl_args)

            else:
                A = self.dynObj.get_state_mat(tt, *state_args)
                B = self.dynObj.get_input_mat(tt, *ctrl_args)

        else:
            raise NotImplementedError("Need to implement this case")

        return A, B

    def prop_state_backward(
        self, tt, x_hat, u_hat, state_args, ctrl_args, inv_state_args, inv_ctrl_args
    ):
        """Propagate the state backward and get the forward state space.

        Parameters
        ----------
        tt : float
            Future timestep.
        x_hat : N x 1 numpy array
            Future state.
        u_hat : Nu x 1 numpy array
            Future control input.
        state_args : tuple
            Extra arguments for the state matrix calculation.
        ctrl_args : tuple
            Extra arguments for the control matrix calculation.
        inv_state_args : tuple
            Extra arguments for the inverse state matrix calculation.
        inv_ctrl_args : tuple
            Extra arguments for the inverse control matrix calculation.

        Returns
        -------
        x_hat_p : N x 1 numpy array
            Previous state.
        A : N x N numpy array
            Forward state transition matrix.
        B : N x Nu
            Forward input matrix.
        c : N x 1 numpy array
            Forward c vector.
        """
        x_hat_p = self._prop_state(
            tt,
            x_hat,
            u_hat,
            state_args,
            ctrl_args,
            False,
            inv_state_args,
            inv_ctrl_args,
        )

        A, B = self.get_state_space(tt, x_hat_p, u_hat, state_args, ctrl_args)
        c = x_hat - A @ x_hat_p - B @ u_hat

        return x_hat_p, A, B, c

    def _determine_cost_matrices(
        self, tt, itr, x_hat, u_hat, is_initial, is_final, cost_args
    ):
        """Calculate the cost matrices."""
        P = self._P
        if is_final:
            Q = self._Q
            q = -(Q @ self._end_state)
            R = np.zeros(self._R.shape)
            r = np.zeros(self.u_nom.shape)
        else:
            if is_initial:
                Q = self._Q
                q = -(Q @ self._init_state)
                R = self._R
                r = -(R @ self.u_nom)
            else:
                Q = np.zeros(self._Q.shape)
                q = np.zeros(self._init_state.shape)
                R = self._R
                r = -(R @ self.u_nom)
        return P, Q, R, q, r

    def _back_pass_update_traj(self, x_hat_p, kk):
        """Get the next state for backward pass, helpful for inherited classes."""
        return x_hat_p.ravel()

    def backward_pass(
        self,
        itr,
        num_timesteps,
        traj,
        state_args,
        ctrl_args,
        cost_args,
        time_vec,
        inv_state_args,
        inv_ctrl_args,
    ):
        """Backward pass for the finite horizon case.

        Parameters
        ----------
        itr : int
            iteration number.
        num_timesteps : int
            Total number of timesteps.
        traj : Nh+1 x N numpy array
            State trajectory.
        state_args : tuple
            Extra arguments for the state matrix calculation.
        ctrl_args : tuple
            Extra arguments for the control matrix calculation.
        cost_args : tuple
            Extra arguments for the cost function. Not used for LQR.
        time_vec : Nh+1 numpy array
            Time vector for control horizon.
        inv_state_args : tuple
            Extra arguments for the inverse state matrix calculation.
        inv_ctrl_args : tuple
            Extra arguments for the inverse control matrix calculation.

        Returns
        -------
        traj : Nh+1 x N numpy array
            State trajectory.
        """
        for kk in range(num_timesteps - 1, -1, -1):
            tt = time_vec[kk]
            u_hat = (
                self.feedback_gain[kk] @ traj[kk + 1, :].reshape((-1, 1))
                + self.feedthrough_gain[kk]
            )
            x_hat_p, A, B, c = self.prop_state_backward(
                tt,
                traj[kk + 1, :].reshape((-1, 1)),
                u_hat,
                state_args,
                ctrl_args,
                inv_state_args,
                inv_ctrl_args,
            )

            P, Q, R, q, r = self._determine_cost_matrices(
                tt, itr, x_hat_p, u_hat, kk == 0, False, cost_args
            )

            ctm_A = self.ct_go_mats[kk + 1] @ A
            ctv_ctm_c = self.ct_go_vecs[kk + 1] + self.ct_go_mats[kk + 1] @ c
            C = P + B.T @ ctm_A
            D = Q + A.T @ ctm_A
            E = R + B.T @ self.ct_go_mats[kk + 1] @ B
            d = q + A.T @ ctv_ctm_c
            e = r + B.T @ ctv_ctm_c

            neg_inv_E = -np.linalg.inv(E)
            self.feedback_gain[kk] = neg_inv_E @ C
            self.feedthrough_gain[kk] = neg_inv_E @ e

            self.ct_go_mats[kk] = D + C.T @ self.feedback_gain[kk]
            self.ct_go_vecs[kk] = d + C.T @ self.feedthrough_gain[kk]

            traj[kk, :] = self._back_pass_update_traj(x_hat_p, kk)

        return traj

    def cost_function(self, tt, state, ctrl_input, is_initial=False, is_final=False):
        """Calculates the cost for the state and control input.

        Parameters
        ----------
        tt : float
            Current timestep.
        state : N x 1 numpy array
            Current state.
        ctrl_input : Nu x 1 numpy array
            Current control input.
        is_initial : bool, optional
            Flag indicating if it's the initial time. The default is False.
        is_final : bool, optional
            Flag indicating if it's the final time. The default is False.

        Returns
        -------
        float
            Cost.
        """
        if is_final:
            sdiff = state - self._end_state
            return (sdiff.T @ self._Q @ sdiff).item()

        else:
            cost = 0
            if is_initial:
                sdiff = state - self._init_state
                cost += (sdiff.T @ self._Q @ sdiff).item()

            cdiff = ctrl_input - self.u_nom
            cost += (cdiff.T @ self._R @ cdiff).item()
        return cost

    def calculate_control(
        self,
        cur_time,
        cur_state,
        end_state=None,
        end_state_tol=1e-2,
        check_inds=None,
        state_args=None,
        ctrl_args=None,
        inv_state_args=None,
        inv_ctrl_args=None,
        provide_details=False,
    ):
        """Calculate the control parameters and state trajectory.

        This can track a reference or regulate to zero. It can also be inifinte
        or finite horizon.

        Parameters
        ----------
        cur_time : float
            Current time when starting the control calculation.
        cur_state : N x 1 numpy array
            Current state.
        end_state : N x 1 numpy array, optional
            Desired/reference ending state. The default is None.
        end_state_tol : float, optional
            Tolerance on the reference state when calculating the state trajectory
            for the inifinte horizon case. The default is 1e-2.
        check_inds : list, optional
            List of indices of the state vector to check when determining
            the end condition for the state trajectory calculation for the
            infinite horizon case. The default is None which checks the full
            state vector.
        state_args : tuple, optional
            Extra arguments for calculating the state matrix. The default is None.
        ctrl_args : tuple, optional
            Extra arguments for calculating the input matrix. The default is None.
        inv_state_args : tuple, optional
            Extra arguments for calculating the inverse state matrix. The
            default is None.
        inv_ctrl_args : tuple, optional
            Extra arguments for calculating the inverse input matrix. The
            default is None.
        provide_details : bool, optional
            Flag indicating if additional outputs should be provided. The
            default is False.

        Returns
        -------
        u : Nu x 1 numpy array
            Control input for current timestep.
        cost : float, optional
            Cost of the trajectory
        state_traj : Nh+1 x N numpy array, optional
            State trajectory over the horizon, or until the reference is reached.
        ctrl_signal : Nh x Nu numy array, optional
            Control inputs over the horizon, or until the reference is reached.
        """
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        if inv_state_args is None:
            inv_state_args = ()
        if inv_ctrl_args is None:
            inv_ctrl_args = ()
        if end_state is None:
            end_state = np.zeros((cur_state.size, 1))

        self._init_state = cur_state.reshape((-1, 1)).copy()
        self._end_state = end_state.reshape((-1, 1)).copy()

        if np.isinf(self.time_horizon) or self.time_horizon <= 0:
            F, G = self.get_state_space(
                cur_time, cur_state, self.u_nom, state_args, ctrl_args
            )
            S = la.solve_discrete_are(F, G, self._Q, self._R)
            self.feedback_gain = la.inv(G.T @ S @ G + self._R) @ (G.T @ S @ F + self._P)
            self.feedthrough_gain = self.u_nom.copy()
            state_traj = cur_state.reshape((1, -1)).copy()

            dx = end_state - cur_state

            if self.dt is None:
                ctrl_signal = (self.feedback_gain @ dx + self.feedthrough_gain).ravel()
                cost = np.nan

            else:
                ctrl_signal = None
                cost = self.cost_function(
                    cur_time, cur_state, self.u_nom, is_initial=True, is_final=False
                )

                if check_inds is None:
                    check_inds = range(cur_state.size)

                timestep = cur_time
                done = (
                    np.linalg.norm(
                        state_traj[-1, check_inds] - end_state[check_inds, 0]
                    )
                    <= end_state_tol
                )

                while not done:
                    timestep += self.dt
                    dx = end_state - state_traj[-1, :].reshape((-1, 1))
                    u = self.feedback_gain @ dx + self.feedthrough_gain
                    if ctrl_signal is None:
                        ctrl_signal = u.flatten()
                    else:
                        ctrl_signal = np.vstack((ctrl_signal, u.ravel()))

                    x = self._prop_state(
                        timestep,
                        state_traj[-1, :].reshape((-1, 1)),
                        u,
                        state_args,
                        ctrl_args,
                        True,
                        (),
                        (),
                    )
                    state_traj = np.vstack((state_traj, x.ravel()))

                    done = (
                        np.linalg.norm(
                            state_traj[-1, check_inds] - end_state[check_inds, 0]
                        )
                        <= end_state_tol
                    )

                    cost += self.cost_function(
                        timestep, x, u, is_initial=False, is_final=done
                    )

        else:
            num_timesteps = int(self.time_horizon / self.dt)
            time_vec = cur_time + self.dt * np.linspace(
                0, 1, num_timesteps + 1, endpoint=True
            )

            self.ct_go_mats = np.zeros(
                (num_timesteps + 1, cur_state.size, cur_state.size)
            )
            self.ct_go_vecs = np.zeros((num_timesteps + 1, cur_state.size, 1))
            self.ct_go_mats[-1] = self._Q.copy()
            self.ct_go_vecs[-1] = -(self._Q @ self._end_state)
            self.feedback_gain = np.zeros(
                (num_timesteps + 1, self.u_nom.size, cur_state.size)
            )
            self.feedthrough_gain = self.u_nom * np.ones(
                (num_timesteps + 1, self.u_nom.size, 1)
            )

            traj = np.nan * np.ones((num_timesteps + 1, cur_state.size))
            traj[-1, :] = end_state.flatten()
            self.backward_pass(
                0,
                num_timesteps,
                traj,
                state_args,
                ctrl_args,
                (),
                time_vec,
                inv_state_args,
                inv_ctrl_args,
            )

            ctrl_signal = np.nan * np.ones((num_timesteps, self.u_nom.size))
            state_traj = np.nan * np.ones((num_timesteps + 1, cur_state.size))
            state_traj[0, :] = cur_state.flatten()
            cost = 0
            for kk, tt in enumerate(time_vec[:-1]):
                u = (
                    self.feedback_gain[kk] @ state_traj[kk, :].reshape((-1, 1))
                    + self.feedthrough_gain[kk]
                )
                ctrl_signal[kk, :] = u.ravel()

                x = self._prop_state(
                    tt,
                    state_traj[kk, :].reshape((-1, 1)),
                    u,
                    state_args,
                    ctrl_args,
                    True,
                    (),
                    (),
                )
                state_traj[kk + 1, :] = x.ravel()

                cost += self.cost_function(
                    tt, x, u, is_initial=False, is_final=(kk == time_vec.size)
                )

        u = ctrl_signal[0, :].reshape((-1, 1))
        details = (
            cost,
            state_traj,
            ctrl_signal,
        )
        return (u, *details) if provide_details else u


class ELQR(LQR):
    """Implements an Extended Linear Quadratic Regulator (ELQR) controller.

    Attributes
    ----------
    max_iters : int
        Maximum number of iterations to try for convergence.
    tol : float
        Tolerance for convergence.
    ct_come_mats : Nh+1 x N x N numpy array
        Cost-to-come matrices, 1 per step in the time horizon.
    ct_come_vecs : Nh+1 x N x 1 numpy array
        Cost-to-come vectors, 1 per step in the time horizon.
    use_custom_cost : bool
        Flag indicating if a custom cost function should be used.
    """

    def __init__(self, max_iters=1e3, tol=1e-4, **kwargs):
        """Initialize an object.

        Parameters
        ----------
        max_iters : int, optional
            Maximum number of iterations to try for convergence. The default is
            1e3.
        tol : flaot, optional
            Tolerance on convergence. The default is 1e-4.
        **kwargs : dict
            Additional arguments.
        """
        key = "time_horizon"
        if key not in kwargs:
            kwargs.update(key, 10)

        super().__init__(**kwargs)
        self.max_iters = int(max_iters)
        self.tol = tol

        self.ct_come_mats = np.array([])
        self.ct_come_vecs = np.array([])

        self.use_custom_cost = False
        self._non_quad_fun = None
        self._quad_modifier = None
        self._cost_fun = None

    def set_cost_model(
        self, Q=None, R=None, non_quadratic_fun=None, quad_modifier=None, cost_fun=None
    ):
        r"""Sets the cost model.

        Either `Q`, `R`, and `non_quadratic_fun` must be supplied (and
        optionally `quad_modifier`) or `cost_fun`.

        Notes
        -----
        This assumes the following form for the cost function

        .. math::

            J = x_f^T Q x_f + \int_{t_0}^{t_f} x^T Q x + u^T R u + u^T P x + f(\tau, x, u) d\tau

        where :math:`f(t, x, u)` contains the non-quadratic terms and gets
        automatically quadratized by the algorithm.

        A custom cost function can also be provided for :math:`J` in which case
        the entire function :math:`J(t, x, u)` will be quadratized at every
        timestep.

        Parameters
        ----------
        Q : N x N numpy array, optional
            State cost matrix. The default is None.
        R : Nu x Nu numpy array, optional
            Control cost matrix. The default is None.
        non_quadratic_fun : callable, optional
            Non-quadratic portion of the standard cost function. This should
            have the form
            :code:`f(t, state, ctrl_input, end_state, is_initial, is_final, *args)`
            and return a scalar additional cost. The default is None.
        quad_modifier : callable, optional
            Function to modifiy the :math:`P, Q, R, q, r` matrices before
            adding the non-quadratic terms. Must have the form
            :code:`f(itr, t, P, Q, R, q, r)` and return a tuple of
            :code:`(P, Q, R, q, r)`. The default is None.
        cost_fun : callable, optional
            Custom cost function, must handle all cases and will be numericaly
            quadratized at every timestep. Must have the form
            :code:`f(t, state, ctrl_input, end_state, is_initial, is_final, *args)`
            and return a scalar total cost. The default is None.

        Raises
        ------
        RuntimeError
            Invlaid combination of input arguments.
        """
        if Q is not None and R is not None and non_quadratic_fun is not None:
            super().set_cost_model(Q, R)
            self._non_quad_fun = non_quadratic_fun
            self._quad_modifier = quad_modifier
            self.use_custom_cost = False

        elif cost_fun is not None:
            self._cost_fun = cost_fun
            self.use_custom_cost = True

        else:
            raise RuntimeError("Invalid combination of inputs.")

    def cost_function(
        self, tt, state, ctrl_input, cost_args, is_initial=False, is_final=False
    ):
        """Calculates the cost for the state and control input.

        Parameters
        ----------
        tt : float
            Current timestep.
        state : N x 1 numpy array
            Current state.
        ctrl_input : Nu x 1 numpy array
            Current control input.
        cost_args : tuple
            Additional arguments for the non-quadratic part or the custom cost
            function.
        is_initial : bool, optional
            Flag indicating if it's the initial time. The default is False.
        is_final : bool, optional
            Flag indicating if it's the final time. The default is False.

        Returns
        -------
        float
            Cost.
        """
        if not self.use_custom_cost:
            cost = super().cost_function(
                tt, state, ctrl_input, is_initial=is_initial, is_final=is_final
            )
            return (
                cost
                if is_final
                else cost
                + self._non_quad_fun(
                    tt,
                    state,
                    ctrl_input,
                    self._end_state,
                    is_initial,
                    is_final,
                    *cost_args
                )
            )

        return self._cost_fun(
            tt, state, ctrl_input, self._end_state, is_initial, is_final, *cost_args
        )

    def prop_state_forward(
        self, tt, x_hat, u_hat, state_args, ctrl_args, inv_state_args, inv_ctrl_args
    ):
        """Propagate the state forward and get the backward state space.

        Parameters
        ----------
        tt : float
            Current timestep.
        x_hat : N x 1 numpy array
            Current state.
        u_hat : Nu x 1 numpy array
            Current control input.
        state_args : tuple
            Extra arguments for the state matrix calculation.
        ctrl_args : tuple
            Extra arguments for the control matrix calculation.
        inv_state_args : tuple
            Extra arguments for the inverse state matrix calculation.
        inv_ctrl_args : tuple
            Extra arguments for the inverse control matrix calculation.

        Returns
        -------
        x_hat_p : N x 1 numpy array
            Next state.
        Abar : N x N numpy array
            Backward state transition matrix.
        Bbar : N x Nu
            Backward input matrix.
        cbar : N x 1 numpy array
            Backward c vector.
        """
        x_hat_p = self._prop_state(
            tt, x_hat, u_hat, state_args, ctrl_args, True, inv_state_args, inv_ctrl_args
        )

        if self.dynObj is not None:
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                if self.dynObj.dt > 0:
                    self.dynObj.dt *= -1  # set to inverse dynamics

                ABar = self.dynObj.get_state_mat(
                    tt, x_hat_p, *state_args, u=u_hat, ctrl_args=ctrl_args
                )
                BBar = self.dynObj.get_input_mat(tt, x_hat_p, u_hat, *ctrl_args)

            else:
                raise NotImplementedError("Need to implement this")

        else:
            raise NotImplementedError("Need to implement this")
        cBar = x_hat - ABar @ x_hat_p - BBar @ u_hat

        return x_hat_p, ABar, BBar, cBar

    def quadratize_cost(self, tt, itr, x_hat, u_hat, is_initial, is_final, cost_args):
        """Quadratizes the cost function.

        If the non-quadratic portion of the standard cost function is not
        positive semi-definite then it is regularized by setting the negative
        eigen values to zero and reconstructing the matrix.

        Parameters
        ----------
        tt : float
            Current timestep.
        itr : int
            Iteration number.
        x_hat : N x 1 numpy array
            Current state.
        u_hat : Nu x 1 numpy array
            Current control input.
        is_initial : bool
            Flag indicating if this is the first timestep.
        is_final : bool
            Flag indicating if this is the last timestep.
        cost_args : tuple
            Additional arguments for the cost calculation.

        Raises
        ------
        NotImplementedError
            Unsupported configuration.

        Returns
        -------
        P : Nu x N numpy array
            State and control correlation cost matrix.
        Q : N x N numpy array
            State cost matrix.
        R : Nu x Nu numpy array
            Control input cost matrix.
        q : N x 1
            State cost vector.
        r : Nu x 1
            Control input cost vector.
        """
        if not self.use_custom_cost:
            P, Q, R, q, r = super()._determine_cost_matrices(
                tt, itr, x_hat, u_hat, is_initial, is_final, cost_args
            )

            if not is_initial and not is_final:
                if self._quad_modifier is not None:
                    P, Q, R, q, r = self._quad_modifier(itr, tt, P, Q, R, q, r)

                xdim = x_hat.size
                udim = u_hat.size
                comb_state = np.vstack((x_hat, u_hat)).ravel()
                big_mat = gmath.get_hessian(
                    comb_state,
                    lambda _x, *_args: self._non_quad_fun(
                        tt,
                        _x[:xdim],
                        _x[xdim:],
                        self._end_state,
                        is_initial,
                        is_final,
                        *cost_args
                    ),
                )

                # regularize hessian to keep it pos semi def
                vals, vecs = np.linalg.eig(big_mat)
                vals[vals < 0] = 0
                big_mat = vecs @ np.diag(vals) @ vecs.T

                # extract non-quadratic terms
                non_Q = big_mat[:xdim, :xdim]

                non_P = big_mat[xdim:, :xdim]
                non_R = big_mat[xdim:, xdim:]

                big_vec = gmath.get_jacobian(
                    comb_state,
                    lambda _x, *args: self._non_quad_fun(
                        tt,
                        _x[:xdim],
                        _x[xdim:],
                        self._end_state,
                        is_initial,
                        is_final,
                        *cost_args
                    ),
                )

                non_q = big_vec[:xdim].reshape((xdim, 1)) - (
                    non_Q @ x_hat + non_P.T @ u_hat
                )
                non_r = big_vec[xdim:].reshape((udim, 1)) - (
                    non_P @ x_hat + non_R @ u_hat
                )

                Q += non_Q
                q += non_q
                R += non_R
                r += non_r
                P += non_P

        else:
            # TODO: get hessians
            raise NotImplementedError("Need to implement this")

        return P, Q, R, q, r

    def _determine_cost_matrices(
        self, tt, itr, x_hat, u_hat, is_initial, is_final, cost_args
    ):
        return self.quadratize_cost(
            tt, itr, x_hat, u_hat, is_initial, is_final, cost_args
        )

    def _back_pass_update_traj(self, x_hat_p, kk):
        return -(
            np.linalg.inv(self.ct_go_mats[kk] + self.ct_come_mats[kk])
            @ (self.ct_go_vecs[kk] + self.ct_come_vecs[kk])
        ).ravel()

    def forward_pass(
        self,
        itr,
        num_timesteps,
        traj,
        state_args,
        ctrl_args,
        cost_args,
        time_vec,
        inv_state_args,
        inv_ctrl_args,
    ):
        """Forward pass for the smoothing.

        Parameters
        ----------
        itr : int
            iteration number.
        num_timesteps : int
            Total number of timesteps.
        traj : Nh+1 x N numpy array
            State trajectory.
        state_args : tuple
            Extra arguments for the state matrix calculation.
        ctrl_args : tuple
            Extra arguments for the control matrix calculation.
        cost_args : tuple
            Extra arguments for the cost function.
        time_vec : Nh+1 numpy array
            Time vector for control horizon.
        inv_state_args : tuple
            Extra arguments for the inverse state matrix calculation.
        inv_ctrl_args : tuple
            Extra arguments for the inverse control matrix calculation.

        Returns
        -------
        traj : Nh+1 x N numpy array
            State trajectory.
        """
        for kk in range(num_timesteps):
            tt = time_vec[kk]

            u_hat = (
                self.feedback_gain[kk] @ traj[kk, :].reshape((-1, 1))
                + self.feedthrough_gain[kk]
            )
            x_hat_p, ABar, BBar, cBar = self.prop_state_forward(
                tt,
                traj[kk, :].reshape((-1, 1)),
                u_hat,
                state_args,
                ctrl_args,
                inv_state_args,
                inv_ctrl_args,
            )

            # final cost is handled after the forward pass
            P, Q, R, q, r = self._determine_cost_matrices(
                tt, itr, traj[kk, :].reshape((-1, 1)), u_hat, kk == 0, False, cost_args
            )

            ctm_Q = self.ct_come_mats[kk] + Q
            ctm_Q_A = ctm_Q @ ABar
            ctv_q_ctm_Q_c = self.ct_come_vecs[kk] + q + ctm_Q @ cBar
            CBar = BBar.T @ ctm_Q_A + P @ ABar
            DBar = ABar.T @ ctm_Q_A
            EBar = BBar.T @ ctm_Q @ BBar + R + P @ BBar + BBar.T @ P.T
            dBar = ABar.T @ ctv_q_ctm_Q_c
            eBar = BBar.T @ ctv_q_ctm_Q_c + r + P @ cBar

            neg_inv_EBar = -np.linalg.inv(EBar)
            self.feedback_gain[kk] = neg_inv_EBar @ CBar
            self.feedthrough_gain[kk] = neg_inv_EBar @ eBar

            self.ct_come_mats[kk + 1] = DBar + CBar.T @ self.feedback_gain[kk]
            self.ct_come_vecs[kk + 1] = dBar + CBar.T @ self.feedthrough_gain[kk]

            traj[kk + 1, :] = -(
                np.linalg.inv(self.ct_go_mats[kk + 1] + self.ct_come_mats[kk + 1])
                @ (self.ct_go_vecs[kk + 1] + self.ct_come_vecs[kk + 1])
            ).ravel()

        return traj

    def calculate_control(
        self,
        tt,
        cur_state,
        end_state,
        state_args=None,
        ctrl_args=None,
        cost_args=None,
        inv_state_args=None,
        inv_ctrl_args=None,
        provide_details=False,
        disp=True,
        show_animation=False,
        save_animation=False,
        plt_opts=None,
        ttl=None,
        fig=None,
        plt_inds=None,
    ):
        """Calculate the control parameters and state trajectory.

        Parameters
        ----------
        tt : float
            Current time when starting the control calculation.
        cur_state : N x 1 numpy array
            Current state.
        end_state : N x 1 numpy array
            Desired/reference ending state.
        state_args : tuple, optional
            Extra arguments for calculating the state matrix. The default is None.
        ctrl_args : tuple, optional
            Extra arguments for calculating the input matrix. The default is None.
        cost_args : tuple, optional
            Extra arguments for the cost function, either the non-quadratic
            part or the custom function. The default is None.
        inv_state_args : tuple, optional
            Extra arguments for calculating the inverse state matrix. The
            default is None.
        inv_ctrl_args : tuple, optional
            Extra arguments for calculating the inverse input matrix. The
            default is None.
        provide_details : bool, optional
            Flag indicating if additional outputs should be provided. The
            default is False.
        disp : bool, optional
            Flag indicating if extra text should be displayed during the
            optimization. The default is True.
        show_animation : bool, optional
            Flag indicating if an animation should be shown during the
            optimization. The default is False.
        save_animation : bool, optional
            Flag indicating if the frames of the animation should be saved.
            Requires that the animation is shown. The default is False.
        plt_opts : dict, optional
            Options for the plot, see :func:`gncpy.plotting.init_plotting_opts`.
            The default is None.
        ttl : string, optional
            Title for the plot. The default is None.
        fig : matplotlib figure, optional
            Figure to draw on. If supplied only the end point and paths are
            drawn, the rest of the figure remains unchanged. The default is None
            which creates a new figure and adds a title.
        plt_inds : list, optional
            2 element list of state indices for plotting. The default is None
            which assumes [0, 1].

        Returns
        -------
        u : Nu x 1 numpy array
            Control input for current timestep.
        cost : float, optional
            Cost of the trajectory
        state_traj : Nh+1 x N numpy array, optional
            State trajectory over the horizon, or until the reference is reached.
        ctrl_signal : Nh x Nu numy array, optional
            Control inputs over the horizon, or until the reference is reached.
        fig : matplotlib figure, optional
            Figure handle for the generated plot, None if animation is not shown.
        frame_list : list, optional
            Each element is a PIL.Image corresponding to an animation frame.
        """
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        if cost_args is None:
            cost_args = ()
        if inv_state_args is None:
            inv_state_args = ()
        if inv_ctrl_args is None:
            inv_ctrl_args = ()

        if plt_inds is None:
            plt_inds = [0, 1]

        old_cost = float("inf")
        num_timesteps = int(self.time_horizon / self.dt)
        self._init_state = cur_state.reshape((-1, 1)).copy()
        self._end_state = end_state.reshape((-1, 1)).copy()
        traj = np.nan * np.ones((num_timesteps + 1, cur_state.size))
        traj[0, :] = cur_state.flatten()

        self.ct_come_mats = np.zeros(
            (num_timesteps + 1, cur_state.size, cur_state.size)
        )
        self.ct_come_vecs = np.zeros((num_timesteps + 1, cur_state.size, 1))
        self.ct_go_mats = np.zeros(self.ct_come_mats.shape)
        self.ct_go_vecs = np.zeros(self.ct_come_vecs.shape)

        self.feedback_gain = np.zeros(
            (num_timesteps + 1, self.u_nom.size, cur_state.size)
        )
        self.feedthrough_gain = self.u_nom * np.ones(
            (num_timesteps + 1, self.u_nom.size, 1)
        )

        abs_dt = np.abs(self.dt)

        frame_list = []
        if show_animation:
            if fig is None:
                fig = plt.figure()
                fig.add_subplot(1, 1, 1)
                fig.axes[0].set_aspect("equal", adjustable="box")

                if plt_opts is None:
                    plt_opts = gplot.init_plotting_opts(f_hndl=fig)

                if ttl is None:
                    ttl = "ELQR"

                gplot.set_title_label(fig, 0, plt_opts, ttl=ttl)

                # draw start
                fig.axes[0].scatter(
                    self._init_state[plt_inds[0], 0],
                    self._init_state[plt_inds[1], 0],
                    marker="o",
                    color="g",
                    zorder=1000,
                )
                fig.tight_layout()

            fig.axes[0].scatter(
                self._end_state[plt_inds[0], 0],
                self._end_state[plt_inds[1], 0],
                marker="x",
                color="r",
                zorder=1000,
            )
            plt.pause(0.01)

            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            fig_w, fig_h = fig.canvas.get_width_height()

            # save first frame of animation
            if save_animation:
                with io.BytesIO() as buff:
                    fig.savefig(buff, format="raw")
                    buff.seek(0)
                    img = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                        (fig_h, fig_w, -1)
                    )
                frame_list.append(Image.fromarray(img))

        if disp:
            print("Starting ELQR optimization loop...")

        time_vec = tt + abs_dt * np.linspace(0, 1, num_timesteps + 1, endpoint=True)

        for ii in range(self.max_iters):
            # forward pass
            traj = self.forward_pass(
                ii,
                num_timesteps,
                traj,
                state_args,
                ctrl_args,
                cost_args,
                time_vec,
                inv_state_args,
                inv_ctrl_args,
            )

            # quadratize final cost (ii = num_timesteps)
            u_hat = (
                self.feedback_gain[num_timesteps]
                @ traj[num_timesteps - 1, :].reshape((-1, 1))
                + self.feedthrough_gain[num_timesteps]
            )
            (
                _,
                self.ct_go_mats[num_timesteps],
                _,
                self.ct_go_vecs[num_timesteps],
                _,
            ) = self.quadratize_cost(
                time_vec[-1],
                ii,
                traj[num_timesteps - 1, :].reshape((-1, 1)),
                u_hat,
                False,
                True,
                cost_args,
            )
            traj[num_timesteps, :] = -(
                np.linalg.inv(
                    self.ct_go_mats[num_timesteps] + self.ct_come_mats[num_timesteps]
                )
                @ (self.ct_go_vecs[num_timesteps] + self.ct_come_vecs[num_timesteps])
            ).ravel()

            if show_animation:
                # plot forward pass trajectory
                fig.axes[0].plot(
                    traj[:, plt_inds[0]],
                    traj[:, plt_inds[1]],
                    color=(0.5, 0.5, 0.5),
                    alpha=0.2,
                    zorder=-10,
                )
                plt.pause(0.005)

            # backward pass
            traj = self.backward_pass(
                ii,
                num_timesteps,
                traj,
                state_args,
                ctrl_args,
                cost_args,
                time_vec,
                inv_state_args,
                inv_ctrl_args,
            )

            if show_animation:
                # plot backward pass trajectory
                fig.axes[0].plot(
                    traj[:, plt_inds[0]],
                    traj[:, plt_inds[1]],
                    color=(0.5, 0.5, 0.5),
                    alpha=0.2,
                    zorder=-10,
                )

                plt.pause(0.005)
                if save_animation:
                    with io.BytesIO() as buff:
                        fig.savefig(buff, format="raw")
                        buff.seek(0)
                        img = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                            (fig_h, fig_w, -1)
                        )
                    frame_list.append(Image.fromarray(img))

            # get cost
            cost = 0
            x = traj[0, :].copy().reshape((-1, 1))
            for kk, tt in enumerate(time_vec):
                u = self.feedback_gain[kk] @ x + self.feedthrough_gain[kk]
                cost += self.cost_function(
                    tt,
                    x,
                    u,
                    cost_args,
                    is_initial=(kk == 0),
                    is_final=(kk == num_timesteps),
                )
                x = self._prop_state(
                    tt, x, u, state_args, ctrl_args, True, inv_state_args, inv_ctrl_args
                )

            if disp:
                print("\tIteration: {:3d} Cost: {:10.4f}".format(ii, cost))

            # check for convergence
            if np.abs(old_cost - cost) / cost < self.tol:
                break
            old_cost = cost

        # create outputs and return
        ctrl_signal = np.nan * np.ones((num_timesteps, self.u_nom.size))
        state_traj = np.nan * np.ones((num_timesteps + 1, self._init_state.size))
        cost = 0
        state_traj[0, :] = self._init_state.flatten()
        for kk, tt in enumerate(time_vec[:-1]):
            ctrl_signal[kk, :] = (
                self.feedback_gain[kk] @ state_traj[kk, :].reshape((-1, 1))
                + self.feedthrough_gain[kk]
            ).ravel()
            cost += self.cost_function(
                tt,
                state_traj[kk, :].reshape((-1, 1)),
                ctrl_signal[kk, :].reshape((-1, 1)),
                cost_args,
                is_initial=(kk == 0),
                is_final=False,
            )
            state_traj[kk + 1, :] = self._prop_state(
                tt,
                state_traj[kk, :].reshape((-1, 1)),
                ctrl_signal[kk, :].reshape((-1, 1)),
                state_args,
                ctrl_args,
                True,
                inv_state_args,
                inv_ctrl_args,
            ).ravel()

        cost += self.cost_function(
            time_vec[-1],
            state_traj[num_timesteps, :].reshape((-1, 1)),
            ctrl_signal[num_timesteps - 1, :].reshape((-1, 1)),
            cost_args,
            is_initial=False,
            is_final=True,
        )

        if show_animation:
            fig.axes[0].plot(
                state_traj[:, plt_inds[0]],
                state_traj[:, plt_inds[1]],
                linestyle="-",
                color="g",
            )
            plt.pause(0.001)
            if save_animation:
                with io.BytesIO() as buff:
                    fig.savefig(buff, format="raw")
                    buff.seek(0)
                    img = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                        (fig_h, fig_w, -1)
                    )
                frame_list.append(Image.fromarray(img))

        u = ctrl_signal[0, :].reshape((-1, 1))
        details = (cost, state_traj, ctrl_signal, fig, frame_list)
        return (u, *details) if provide_details else u
