import numpy as np
import scipy.linalg as la

import gncpy.dynamics.basic as gdyn
import gncpy.math as gmath


class BaseLQR:
    r""" Implements a Linear Quadratic Regulator (LQR) controller.

        This implements an LQR controller for the cost function

        .. math::
            J = \frac{1}{2} \left[x_f^T Q x_f + \int^{t_f}_0 x^T Q x + u^T R u
                     + u^T P x\right]

    Args:
        Q (N x N numpy array): State penalty matrix (default: empty array)
        R (Nu x Nu numpy array): Control penalty matrix (default: empty arary)
        cross_penalty (Nu x N numpy array): Cross penalty matrix (default:
            zero)
        horizon_len (int): Length of trajectory to optimize over (default: Inf)

    Attributes:
        state_penalty (N x N numpy array): State penalty matrix :math:`Q`
        ctrl_penalty (Nu x Nu numpy array): Control penalty matrix :math:`R`
        cross_penalty (Nu x N numpy array): Cross penalty matrix
        horizon_len (int): Length of trajectory to optimize over
    """

    def __init__(self, Q=None, R=None, cross_penalty=None, horizon_len=None, **kwargs):
        if Q is None:
            Q = np.array([[]])
        self.state_penalty = Q
        if R is None:
            R = np.array([[]])
        self.ctrl_penalty = R
        if cross_penalty is None:
            def_rows = self.ctrl_penalty.shape[1]
            if self.state_penalty.size > 0:
                def_cols = self.state_penalty.shape[0]
            else:
                def_cols = 0
            cross_penalty = np.zeros((def_rows, def_cols))
        self.cross_penalty = cross_penalty
        if horizon_len is None:
            horizon_len = np.inf
        self.horizon_len = horizon_len
        super().__init__(**kwargs)

    def iterate(self, F, G, **kwargs):
        """Calculates the feedback gain.

        If using a finite time horizon, this loops over the entire horizon
        to calculate the gain :math:`K` such that the control input is

        .. math::
            u = -Kx

        Args:
            F (N x N numpy array): Discrete time state matrix
            G (N x Nu numpy array): Discrete time input matrix

        Raises:
            RuntimeError: Raised for the finite horizon case

        Todo:
            Implement the finite horizon case

        Returns:
            (Nu x N numpy array): Feedback gain :math:`K`
        """

        if self.horizon_len == np.inf:
            P = la.solve_discrete_are(F, G, self.state_penalty, self.ctrl_penalty)
            feedback_gain = la.inv(G.T @ P @ G + self.ctrl_penalty) @ (
                G.T @ P @ F + self.cross_penalty
            )
            return feedback_gain
        else:
            # ##TODO: implement
            c = self.__class__.__name__
            name = self.iterate.__name__
            msg = "{}.{} not implemented".format(c, name)
            raise RuntimeError(msg)


class ELQR:
    def __init__(self, max_iters=1e3, tol=1e-4, time_horizon=10):
        self.max_iters = int(max_iters)
        self.tol = tol
        self.time_horizon = time_horizon
        self._dt = None
        self.start_time = None

        self.u_nom = np.array([])
        self.ct_come_mats = np.array([])
        self.ct_come_vecs = np.array([])
        self.ct_go_mats = np.array([])
        self.ct_go_vecs = np.array([])
        self.feedback_gain = np.array([])
        self.feedthrough_gain = np.array([])

        self.use_custom_cost = False
        self._init_state = np.array([])
        self._end_state = np.array([])
        self._Q = None
        self._R = None
        self._non_quad_fun = None
        self._quad_modifier = None
        self._cost_fun = None

        self.dynObj = None

    @property
    def dt(self):
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
        self.u_nom = u_nom.reshape((-1, 1))
        self.dynObj = dynObj
        if dt is not None:
            self.dt = dt

    def set_cost_model(
        self, Q=None, R=None, non_quadratic_fun=None, quad_modifier=None, cost_fun=None
    ):
        if Q is not None and R is not None and non_quadratic_fun is not None:
            self._Q = Q
            self._R = R
            self._non_quad_fun = non_quadratic_fun
            self._quad_modifier = quad_modifier
        elif cost_fun is not None:
            self._cost_fun = cost_fun
            self.use_custom_cost = True

        else:
            raise RuntimeError("Invalid combination of inputs.")

    def cost_function(
        self, tt, state, ctrl_input, cost_args, is_initial=False, is_final=False,
    ):
        if not self.use_custom_cost:
            if is_final:
                sdiff = state - self._end_state
                return 0.5 * (sdiff.T @ self._Q @ sdiff).item()
            else:
                cost = 0
                if is_initial:
                    sdiff = state - self._init_state
                    cost += 0.5 * (sdiff.T @ self._Q @ sdiff).item()

                cdiff = ctrl_input = self.u_nom
                cost += 0.5 * (cdiff.T @ self._R @ cdiff).item()
                cost += self._non_quad_fun(
                    tt,
                    state,
                    ctrl_input,
                    self._end_state,
                    is_initial,
                    is_final,
                    *cost_args
                )

                return cost

        return self._cost_fun(
            tt, state, ctrl_input, self._end_state, is_initial, is_final, *cost_args
        )

    def prop_state_forward(self, tt, x_hat, u_hat, state_args, ctrl_args):
        if self.dynObj is not None:
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                if self.dynObj.dt < 0:
                    self.dynObj.dt *= -1  # set to go forward
                x_hat_p = self.dynObj.propagate_state(
                    tt, x_hat, u=u_hat, state_args=state_args, ctrl_args=ctrl_args
                )

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

    def prop_state_backward(self, tt, x_hat, u_hat, state_args, ctrl_args):
        if self.dynObj is not None:
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                if self.dynObj.dt > 0:
                    self.dynObj.dt *= -1  # set to go backward
                x_hat_p = self.dynObj.propagate_state(
                    tt, x_hat, u=u_hat, state_args=state_args, ctrl_args=ctrl_args
                )
                self.dynObj.dt *= -1  # flip back to forward to get forward matrices
                A = self.dynObj.get_state_mat(
                    tt, x_hat_p, *state_args, u=u_hat, ctrl_args=ctrl_args
                )
                B = self.dynObj.get_input_mat(tt, x_hat_p, u_hat, *ctrl_args)

            else:
                raise NotImplementedError("Need to implement this case")

        else:
            raise NotImplementedError("Need to implement this case")

        c = x_hat - A @ x_hat_p - B @ u_hat

        return x_hat_p, A, B, c

    def quadratize_cost(self, tt, itr, x_hat, u_hat, is_initial, is_final, cost_args):
        if not self.use_custom_cost:
            P = np.zeros((self.u_nom.size, x_hat.size))
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

                # regularize hessian to keep it pos semi def
                # vals, vecs = np.linalg.eig(non_Q)
                # vals[vals < 0] = 0
                # non_Q = vecs @ np.diag(vals) @ vecs.T

                non_P = big_mat[xdim:, :xdim]
                non_R = big_mat[xdim:, xdim:]

                # regularize hessian to keep it pos semi def
                # vals, vecs = np.linalg.eig(non_R)
                # vals[vals < 0] = 0
                # non_R = vecs @ np.diag(vals) @ vecs.T

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

        else:
            # TODO: get hessians
            raise NotImplementedError("Need to implement this")

        return P, Q, R, q, r

    def forward_pass(self, itr, num_timesteps, x_hat, state_args, ctrl_args, cost_args):
        for kk in range(num_timesteps):
            tt = kk * abs(self.dt) + self.start_time

            u_hat = self.feedback_gain[kk] @ x_hat + self.feedthrough_gain[kk]
            x_hat_p, ABar, BBar, cBar = self.prop_state_forward(
                tt, x_hat, u_hat, state_args, ctrl_args
            )

            # final cost is handled after the forward pass
            P, Q, R, q, r = self.quadratize_cost(
                tt, itr, x_hat, u_hat, kk == 0, False, cost_args
            )

            CBar = BBar.T @ (self.ct_come_mats[kk] + Q) @ ABar + P @ ABar
            DBar = ABar.T @ (self.ct_come_mats[kk] + Q) @ ABar
            EBar = (
                BBar.T @ (self.ct_come_mats[kk] + Q) @ BBar
                + R
                + P @ BBar
                + BBar.T @ P.T
            )
            dBar = ABar.T @ (
                self.ct_come_vecs[kk] + q + (self.ct_come_mats[kk] + Q) @ cBar
            )
            eBar = (
                BBar.T
                @ (self.ct_come_vecs[kk] + q + (self.ct_come_mats[kk] + Q) @ cBar)
                + r
                + P @ cBar
            )

            self.feedback_gain[kk] = -np.linalg.inv(EBar) @ CBar
            self.feedthrough_gain[kk] = -np.linalg.inv(EBar) @ eBar

            self.ct_come_mats[kk + 1] = DBar + CBar.T @ self.feedback_gain[kk]
            self.ct_come_vecs[kk + 1] = dBar + CBar.T @ self.feedthrough_gain[kk]

            x_hat = -(
                np.linalg.inv(self.ct_go_mats[kk + 1] + self.ct_come_mats[kk + 1])
                @ (self.ct_go_vecs[kk + 1] + self.ct_come_vecs[kk + 1])
            )

        return x_hat

    def backward_pass(
        self, itr, num_timesteps, x_hat, state_args, ctrl_args, cost_args
    ):
        for kk in range(num_timesteps - 1, -1, -1):
            tt = kk * abs(self.dt) + self.start_time

            u_hat = self.feedback_gain[kk] @ x_hat + self.feedthrough_gain[kk]
            x_hat_p, A, B, c = self.prop_state_backward(
                tt, x_hat, u_hat, state_args, ctrl_args
            )

            P, Q, R, q, r = self.quadratize_cost(
                tt, itr, x_hat_p, u_hat, kk == 0, False, cost_args
            )

            C = P + B.T @ self.ct_go_mats[kk + 1] @ A
            D = Q + A.T @ self.ct_go_mats[kk + 1] @ A
            E = R + B.T @ self.ct_go_mats[kk + 1] @ B
            d = q + A.T @ (self.ct_go_vecs[kk + 1] + self.ct_go_mats[kk + 1] @ c)
            e = r + B.T @ (self.ct_go_vecs[kk + 1] + self.ct_go_mats[kk + 1] @ c)

            self.feedback_gain[kk] = -np.linalg.inv(E) @ C
            self.feedthrough_gain[kk] = -np.linalg.inv(E) @ e

            self.ct_go_mats[kk] = D + C.T @ self.feedback_gain[kk]
            self.ct_go_vecs[kk] = d + C.T @ self.feedthrough_gain[kk]

            x_hat = -(
                np.linalg.inv(self.ct_go_mats[kk] + self.ct_come_mats[kk])
                @ (self.ct_go_vecs[kk] + self.ct_come_vecs[kk])
            )

        return x_hat

    def calculate_control(
        self,
        tt,
        cur_state,
        end_state,
        state_args=None,
        ctrl_args=None,
        cost_args=None,
        provide_details=False,
    ):
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        if cost_args is None:
            cost_args = ()

        self.start_time = tt

        old_cost = float("inf")
        num_timesteps = int(self.time_horizon / self.dt)
        self._init_state = cur_state.reshape((-1, 1))
        self._end_state = end_state.reshape((-1, 1))
        x_hat = cur_state.reshape((-1, 1))

        # TODO: check initialization
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

        for ii in range(self.max_iters):
            # forward pass
            x_hat = self.forward_pass(
                ii, num_timesteps, x_hat, state_args, ctrl_args, cost_args
            )

            # quadratize final cost (ii = num_timesteps)
            tt = num_timesteps * self.dt + self.start_time
            u_hat = (
                self.feedback_gain[num_timesteps] @ x_hat
                + self.feedthrough_gain[num_timesteps]
            )
            (
                _,
                self.ct_go_mats[num_timesteps],
                _,
                self.ct_go_vecs[num_timesteps],
                _,
            ) = self.quadratize_cost(tt, ii, x_hat, u_hat, False, True, cost_args)
            x_hat = -(
                np.linalg.inv(
                    self.ct_go_mats[num_timesteps] + self.ct_come_mats[num_timesteps]
                )
                @ (self.ct_go_vecs[num_timesteps] + self.ct_come_vecs[num_timesteps])
            )

            # backward pass
            x_hat = self.backward_pass(
                ii, num_timesteps, x_hat, state_args, ctrl_args, cost_args
            )

            # get cost
            cost = 0
            x = x_hat.copy()
            for kk in range(num_timesteps + 1):
                tt = kk * abs(self.dt) + self.start_time
                u = self.feedback_gain[kk] @ x + self.feedthrough_gain[kk]
                cost += self.cost_function(
                    tt,
                    x,
                    u,
                    cost_args,
                    is_initial=(kk == 0),
                    is_final=(kk == num_timesteps),
                )
                x = self.prop_state_forward(tt, x, u, state_args, ctrl_args)[0]

            # check for convergence
            if abs(old_cost - cost) / cost < self.tol:
                break
            old_cost = cost

            print("iter ", ii, " cost: ", cost)

        # create outputs and return
        ctrl_signal = np.nan * np.ones((num_timesteps, self.u_nom.size))
        state_traj = np.nan * np.ones((num_timesteps + 1, self._init_state.size))
        cost = 0
        state_traj[0, :] = self._init_state.flatten()
        for kk in range(num_timesteps):
            tt = kk * abs(self.dt) + self.start_time
            ctrl_signal[kk, :] = (
                self.feedback_gain[kk] @ x + self.feedthrough_gain[kk]
            ).ravel()
            cost += self.cost_function(
                tt,
                state_traj[kk, :].reshape((-1, 1)),
                ctrl_signal[kk, :].reshape((-1, 1)),
                cost_args,
                is_initial=(kk == 0),
                is_final=False,
            )
            state_traj[kk + 1, :] = self.prop_state_forward(
                tt,
                state_traj[kk, :].reshape((-1, 1)),
                ctrl_signal[kk, :].reshape((-1, 1)),
                state_args,
                ctrl_args,
            )[0].ravel()

        cost += self.cost_function(
            tt,
            state_traj[num_timesteps, :].reshape((-1, 1)),
            ctrl_signal[num_timesteps - 1, :].reshape((-1, 1)),
            cost_args,
            is_initial=False,
            is_final=True,
        )

        u = ctrl_signal[0, :].reshape((-1, 1))
        details = (cost, state_traj, ctrl_signal)
        return (u, *details) if provide_details else u
