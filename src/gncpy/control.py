import io
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from PIL import Image

import gncpy.dynamics.basic as gdyn
import gncpy.math as gmath
import gncpy.plotting as gplot


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


class LQR:
    r""" Implements a Linear Quadratic Regulator (LQR) controller.

        This implements an LQR controller for the cost function

        .. math::
            J = \frac{1}{2} \left[x_f^T Q x_f + \int^{t_f}_0 x^T Q x + u^T R u
                     + u^T P x\right]
    """

    def __init__(self):
        super().__init__()

        self._Q = None
        self._R = None
        self._P = None

        self.dynObj = None
        self._dt = None

    @property
    def dt(self):
        """Timestep in seconds."""
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

    def set_cost_model(
        self, Q, R, P=None,
    ):
        self._Q = Q
        self._R = R

        if P is None:
            self._P = np.zeros((self._R.shape[0], self._Q.shape[0]))
        else:
            self._P = P

    def calculate_control(
        self,
        cur_time,
        cur_state,
        time_horizon,
        end_state=None,
        end_state_tol=1e-2,
        check_inds=None,
        state_args=None,
        ctrl_args=None,
        provide_details=False,
    ):
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()

        if np.isinf(time_horizon) or time_horizon <= 0:
            F = self.dynObj.get_state_mat(cur_time, *state_args)
            G = self.dynObj.get_input_mat(cur_time, cur_state, *ctrl_args)
            # TODO: make this work for other dynamic models
            # F, G = self.get_state_space(cur_time, cur_state, state_args)
            P = la.solve_discrete_are(F, G, self._Q, self._R)
            feedback_gain = la.inv(G.T @ P @ G + self._R) @ (G.T @ P @ F + self._P)
            state_traj = cur_state.reshape((1, -1)).copy()

            if end_state is not None:
                dx = end_state - cur_state

                if self.dt is None:
                    dx = cur_state
                    ctrl_signal = (feedback_gain @ dx + self.u_nom).ravel()
                    cost = np.nan

                else:
                    ctrl_signal = None
                    cost = 0

                    if check_inds is None:
                        check_inds = range(cur_state.size)

                    timestep = cur_time
                    while (
                        np.linalg.norm(
                            state_traj[-1, check_inds] - end_state[check_inds, 0]
                        )
                        > end_state_tol
                    ):
                        timestep += self.dt
                        dx = end_state - state_traj[-1, :].reshape((-1, 1))
                        u = feedback_gain @ dx + self.u_nom
                        if ctrl_signal is None:
                            ctrl_signal = u.flatten()
                        else:
                            ctrl_signal = np.vstack((ctrl_signal, u.ravel()))

                        # TODO: fix this for other dynamics models
                        x = self.dynObj.propagate_state(
                            timestep,
                            state_traj[-1, :].reshape((-1, 1)),
                            u=u,
                            state_args=state_args,
                        )
                        state_traj = np.vstack((state_traj, x.ravel()))

                        # TODO: update cost here
                        # cost += cost_fun()

            else:
                dx = cur_state
                ctrl_signal = (feedback_gain @ dx + self.u_nom).ravel()
                cost = np.nan

        else:
            feedback_gain = None
            num_timesteps = int(time_horizon / self.dt)
            time_vec = np.arange(
                self.start_time, self.dt * (num_timesteps + 1), self.dt
            )
            ctrl_signal = np.nan * np.ones((num_timesteps, self.u_nom.size))
            state_traj = np.nan * np.ones((num_timesteps + 1, cur_state.size))
            cost = 0
            for kk, tt in enumerate(time_vec):
                # TODO: implement this
                pass

            raise NotImplementedError()

        u = ctrl_signal[0, :].reshape((-1, 1))
        details = (
            cost,
            state_traj,
            ctrl_signal,
            feedback_gain,
        )
        return (u, *details) if provide_details else u


class ELQR(LQR):
    def __init__(self, max_iters=1e3, tol=1e-4, time_horizon=10):
        super().__init__()
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
        self._non_quad_fun = None
        self._quad_modifier = None
        self._cost_fun = None

    def set_cost_model(
        self, Q=None, R=None, non_quadratic_fun=None, quad_modifier=None, cost_fun=None
    ):
        if Q is not None and R is not None and non_quadratic_fun is not None:
            super().set_cost_model(Q, R)
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
                return (sdiff.T @ self._Q @ sdiff).item()
            else:
                cost = 0
                if is_initial:
                    sdiff = state - self._init_state
                    cost += (sdiff.T @ self._Q @ sdiff).item()

                cdiff = ctrl_input - self.u_nom
                cost += (cdiff.T @ self._R @ cdiff).item()
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

    def _prop_state(self, tt, x_hat, u_hat, state_args, ctrl_args, forward):
        if self.dynObj is not None:
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                if forward and self.dynObj.dt < 0:
                    self.dynObj.dt *= -1  # set to go forward
                elif not forward and self.dynObj.dt > 0:
                    self.dynObj.dt *= -1

                return self.dynObj.propagate_state(
                    tt, x_hat, u=u_hat, state_args=state_args, ctrl_args=ctrl_args
                )

    def prop_state_forward(self, tt, x_hat, u_hat, state_args, ctrl_args):
        x_hat_p = self._prop_state(tt, x_hat, u_hat, state_args, ctrl_args, True)

        if self.dynObj is not None:
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                # if self.dynObj.dt > 0:
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
        x_hat_p = self._prop_state(tt, x_hat, u_hat, state_args, ctrl_args, False)

        if self.dynObj is not None:
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                # if self.dynObj.dt < 0:
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

    def forward_pass(
        self, itr, num_timesteps, traj, state_args, ctrl_args, cost_args, time_vec
    ):
        abs_dt = np.abs(self.dt)
        for kk in range(num_timesteps):
            # tt = kk * abs_dt + self.start_time
            tt = time_vec[kk]

            u_hat = (
                self.feedback_gain[kk] @ traj[kk, :].reshape((-1, 1))
                + self.feedthrough_gain[kk]
            )
            x_hat_p, ABar, BBar, cBar = self.prop_state_forward(
                tt, traj[kk, :].reshape((-1, 1)), u_hat, state_args, ctrl_args
            )

            # final cost is handled after the forward pass
            P, Q, R, q, r = self.quadratize_cost(
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

    def backward_pass(
        self, itr, num_timesteps, traj, state_args, ctrl_args, cost_args, time_vec
    ):
        abs_dt = np.abs(self.dt)
        for kk in range(num_timesteps - 1, -1, -1):
            # tt = kk * abs_dt + self.start_time
            tt = time_vec[kk]

            u_hat = (
                self.feedback_gain[kk] @ traj[kk + 1, :].reshape((-1, 1))
                + self.feedthrough_gain[kk]
            )
            x_hat_p, A, B, c = self.prop_state_backward(
                tt, traj[kk + 1, :].reshape((-1, 1)), u_hat, state_args, ctrl_args
            )

            P, Q, R, q, r = self.quadratize_cost(
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

            traj[kk, :] = -(
                np.linalg.inv(self.ct_go_mats[kk] + self.ct_come_mats[kk])
                @ (self.ct_go_vecs[kk] + self.ct_come_vecs[kk])
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
        provide_details=False,
        disp=True,
        show_animation=False,
        save_animation=False,
        plt_opts=None,
        ttl=None,
        fig=None,
        plt_inds=None,
    ):
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        if cost_args is None:
            cost_args = ()

        if plt_inds is None:
            plt_inds = [0, 1]

        self.start_time = tt

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

        time_vec = np.arange(self.start_time, abs_dt * (num_timesteps + 1), abs_dt)

        for ii in range(self.max_iters):
            # forward pass
            traj = self.forward_pass(
                ii, num_timesteps, traj, state_args, ctrl_args, cost_args, time_vec
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
                ii, num_timesteps, traj, state_args, ctrl_args, cost_args, time_vec
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
                x = self._prop_state(tt, x, u, state_args, ctrl_args, True)

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
            # tt = kk * abs_dt + self.start_time
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
