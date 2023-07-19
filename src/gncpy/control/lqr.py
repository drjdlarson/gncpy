import numpy as np
import scipy.linalg as la

import gncpy.dynamics.basic as gdyn


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
    end_state : N x 1
        Ending state. This generally does not need to be set directly.
    hard_constraints : bool
            Flag indicating that state constraints should be enforced during value propagation.
    """

    def __init__(self, time_horizon=float("inf"), hard_constraints: bool = False):
        """Initialize an object.

        Parameters
        ----------
        time_horizon : float, optional
            Time horizon for the controller. The default is float("inf").
        hard_constraints : bool, optional
            Flag indicating that state constraints should be enforced during value propagation.
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
        self.end_state = np.array([])
        self.hard_constraints = hard_constraints
        self.control_constraints = None

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

    @property
    def Q(self):
        """Read only state penalty matrix."""
        return self._Q

    @property
    def R(self):
        """Read only control penalty matrix."""
        return self._R

    def set_state_model(self, u_nom, control_constraints=None, dynObj=None, dt=None):
        """Set the state/dynamics model.

        Parameters
        ----------
        u_nom : Nu x 1 numpy array
            Nominal control input.
        control_constraints : callable
            Function that takes in timestep and control signal and returns the constrained control signal as a Nu x 1
            numpy array
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
        if control_constraints is not None:
            self.control_constraints = control_constraints

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
        if Q.shape[0] != Q.shape[1]:
            raise RuntimeError("Q must b a square matrix!")
        if R.shape[0] != R.shape[1]:
            raise RuntimeError("R must b a square matrix!")
        self._Q = Q
        self._R = R

        if P is None:
            self._P = np.zeros((self._R.shape[0], self._Q.shape[0]))
        else:
            self._P = P

    def prop_state(
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
                if forward:
                    return self.dynObj.propagate_state(
                        tt, x_hat, u=u_hat, state_args=state_args, ctrl_args=ctrl_args
                    )
                else:
                    A, B = self.get_state_space(tt, x_hat, u_hat, state_args, ctrl_args)
                    prev_state = la.inv(A) @ (x_hat - B @ u_hat)
                    if self.dynObj.state_constraint is not None:
                        prev_state = self.dynObj.state_constraint(tt, prev_state)
                    return prev_state

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
                B = self.dynObj.get_input_mat(
                    tt, x_hat, u_hat, state_args=state_args, ctrl_args=ctrl_args
                )

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
        x_hat_p = self.prop_state(
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
            q = -(Q @ self.end_state)
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

    def backward_pass_step(
        self,
        itr,
        kk,
        time_vec,
        traj,
        state_args,
        ctrl_args,
        cost_args,
        inv_state_args,
        inv_ctrl_args,
    ):
        tt = time_vec[kk]
        u_hat = (
            self.feedback_gain[kk] @ traj[kk + 1, :].reshape((-1, 1))
            + self.feedthrough_gain[kk]
        )
        if self.hard_constraints and self.control_constraints is not None:
            u_hat = self.control_constraints(tt, u_hat)
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

        out = self._back_pass_update_traj(x_hat_p, kk)
        if (
            self.hard_constraints
            and self.dynObj is not None
            and self.dynObj.state_constraint is not None
        ):
            out = self.dynObj.state_constraint(time_vec[kk], out.reshape((-1, 1)))
        return out.ravel()

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
            traj[kk, :] = self.backward_pass_step(
                itr,
                kk,
                time_vec,
                traj,
                state_args,
                ctrl_args,
                cost_args,
                inv_state_args,
                inv_ctrl_args,
            )

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
            sdiff = state - self.end_state
            return (sdiff.T @ self._Q @ sdiff).item()

        else:
            cost = 0
            if is_initial:
                sdiff = state - self._init_state
                cost += (sdiff.T @ self._Q @ sdiff).item()

            cdiff = ctrl_input - self.u_nom
            cost += (cdiff.T @ self._R @ cdiff).item()
        return cost

    def solve_dare(self, cur_time, cur_state, state_args=None, ctrl_args=None):
        """Solve the discrete algebraic ricatti equation.

        Parameters
        ----------
        cur_time : float
            Current time.
        cur_state : N x 1 numpy array
            Current state.
        state_args : tuple, optional
            Additional arguments for calculating the state. The default is None.
        ctrl_args : tuple, optional
            Additional agruments for calculating the input matrix. The default
            is None.

        Returns
        -------
        S : N x N numpy array
            DARE solution.
        F : N x N numpy array
            Discrete state transition matrix
        G : N x Nu numpy array
            Discrete input matrix.
        """
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        F, G = self.get_state_space(
            cur_time, cur_state, self.u_nom, state_args, ctrl_args
        )
        S = la.solve_discrete_are(F, G, self._Q, self._R)

        return S, F, G

    def calculate_control(
        self,
        cur_time,
        cur_state,
        end_state=None,
        end_state_tol=1e-2,
        max_inf_iters=int(1e3),
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
        max_inf_iters : int
            Maximum number of steps to use in the trajectory when finding
            the state trajectory for the infinite horizon case to avoid infinite
            loops.
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
        self.end_state = end_state.reshape((-1, 1)).copy()

        if np.isinf(self.time_horizon) or self.time_horizon <= 0:
            S, F, G = self.solve_dare(
                cur_time, cur_state, state_args=state_args, ctrl_args=ctrl_args
            )
            self.feedback_gain = la.inv(G.T @ S @ G + self._R) @ (G.T @ S @ F + self._P)
            self.feedthrough_gain = self.u_nom.copy()
            state_traj = cur_state.reshape((1, -1)).copy()

            dx = end_state - cur_state

            if self.dt is None:
                ctrl_signal = (self.feedback_gain @ dx + self.feedthrough_gain).ravel()
                if self.control_constraints is not None:
                    ctrl_signal = self.control_constraints(0, ctrl_signal).ravel()
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

                itr = 0
                while not done:
                    timestep += self.dt
                    dx = end_state - state_traj[-1, :].reshape((-1, 1))
                    u = self.feedback_gain @ dx + self.feedthrough_gain
                    if self.control_constraints is not None:
                        u = self.control_constraints(timestep, u)
                    if ctrl_signal is None:
                        ctrl_signal = u.reshape((1, -1))
                    else:
                        ctrl_signal = np.vstack((ctrl_signal, u.ravel()))

                    x = self.prop_state(
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

                    itr += 1
                    done = (
                        np.linalg.norm(
                            state_traj[-1, check_inds] - end_state[check_inds, 0]
                        )
                        <= end_state_tol
                        or itr >= max_inf_iters
                    )

                    cost += self.cost_function(
                        timestep, x, u, is_initial=False, is_final=done
                    )

                # if we start too close to the end state ctrl_signal can be None
                if ctrl_signal is None:
                    ctrl_signal = self.u_nom.copy().reshape((1, -1))

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
            self.ct_go_vecs[-1] = -(self._Q @ self.end_state)
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
                if self.control_constraints is not None:
                    u = self.control_constraints(tt, u)
                ctrl_signal[kk, :] = u.ravel()

                x = self.prop_state(
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
