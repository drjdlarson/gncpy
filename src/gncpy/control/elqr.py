import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from PIL import Image

import gncpy.dynamics.basic as gdyn
import gncpy.math as gmath
import gncpy.plotting as gplot
from gncpy.control.lqr import LQR


class ELQR(LQR):
    """Implements an Extended Linear Quadratic Regulator (ELQR) controller.

    This is based on
    :cite:`Berg2016_ExtendedLQRLocallyOptimalFeedbackControlforSystemswithNonLinearDynamicsandNonQuadraticCost`.

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
    gif_frame_skip : int
        Number of frames to skip when saving the gif. Set to 1 to take every
        frame.
    """

    def __init__(self, max_iters=1e3, tol=1e-4, **kwargs):
        """Initialize an object.

        Parameters
        ----------
        max_iters : int, optional
            Maximum number of iterations to try for convergence. The default is
            1e3.
        tol : float, optional
            Tolerance on convergence. The default is 1e-4.
        **kwargs : dict
            Additional arguments.
        """
        key = "time_horizon"
        if key not in kwargs:
            kwargs[key] = 10

        super().__init__(**kwargs)
        self.max_iters = int(max_iters)
        self.tol = tol

        self.ct_come_mats = np.array([])
        self.ct_come_vecs = np.array([])

        self.use_custom_cost = False
        self.gif_frame_skip = 1
        self._non_quad_fun = None
        self._quad_modifier = None
        self._cost_fun = None

        self._ax = 0

    def set_cost_model(
        self,
        Q=None,
        R=None,
        non_quadratic_fun=None,
        quad_modifier=None,
        cost_fun=None,
        skip_validity_check=False,
    ):
        r"""Sets the cost model.

        Either `Q`, and `R` must be supplied (and optionally `quad_modifier`)
        or `cost_fun`. If `Q and `R` are specified a `non_quadratic_fun` is
        also needed by the code but this function does not force the requirement.

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
        skip_validity_check : bool
            Flag indicating if an error should be raised if the wrong input
            combination is given.

        Raises
        ------
        RuntimeError
            Invlaid combination of input arguments.
        """
        if non_quadratic_fun is not None:
            self._non_quad_fun = non_quadratic_fun

        self._quad_modifier = quad_modifier

        if Q is not None and R is not None:
            super().set_cost_model(Q, R)
            self.use_custom_cost = False

        elif cost_fun is not None:
            self._cost_fun = cost_fun
            self.use_custom_cost = True

        else:
            if not skip_validity_check:
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
                    self.end_state,
                    is_initial,
                    is_final,
                    *cost_args,
                )
            )

        return self._cost_fun(
            tt, state, ctrl_input, self.end_state, is_initial, is_final, *cost_args
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
        x_hat_p = self.prop_state(
            tt, x_hat, u_hat, state_args, ctrl_args, True, inv_state_args, inv_ctrl_args
        )

        if self.dynObj is not None:
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                if self.dynObj.dt > 0:
                    self.dynObj.dt *= -1  # set to inverse dynamics

                ABar = self.dynObj.get_state_mat(
                    tt, x_hat_p, *inv_state_args, u=u_hat, ctrl_args=ctrl_args
                )
                BBar = self.dynObj.get_input_mat(tt, x_hat_p, u_hat, *inv_ctrl_args)

            else:
                A, B = self.get_state_space(tt, x_hat_p, u_hat, state_args, ctrl_args)
                ABar = la.inv(A)
                BBar = -ABar @ B

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
                        self.end_state,
                        is_initial,
                        is_final,
                        *cost_args,
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
                        self.end_state,
                        is_initial,
                        is_final,
                        *cost_args,
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

    def quadratize_final_cost(self, itr, num_timesteps, traj, time_vec, cost_args):
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
            itr,
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

        if (
            self.hard_constraints
            and self.dynObj is not None
            and self.dynObj.state_constraint is not None
        ):
            traj[num_timesteps, :] = self.dynObj.state_constraint(
                time_vec[num_timesteps], traj[num_timesteps, :].reshape((-1, 1))
            ).ravel()

        return traj

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

    def forward_pass_step(
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
            self.feedback_gain[kk] @ traj[kk, :].reshape((-1, 1))
            + self.feedthrough_gain[kk]
        )
        if self.hard_constraints and self.control_constraints is not None:
            u_hat = self.control_constraints(tt, u_hat)

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

        out = -(
            np.linalg.inv(self.ct_go_mats[kk + 1] + self.ct_come_mats[kk + 1])
            @ (self.ct_go_vecs[kk + 1] + self.ct_come_vecs[kk + 1])
        )
        if (
            self.hard_constraints
            and self.dynObj is not None
            and self.dynObj.state_constraint is not None
        ):
            out = self.dynObj.state_constraint(time_vec[kk + 1], out)
        return out.ravel()

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
            traj[kk + 1, :] = self.forward_pass_step(
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

    def draw_traj(
        self,
        fig,
        plt_inds,
        fig_h,
        fig_w,
        save_animation,
        num_timesteps,
        time_vec,
        state_args,
        ctrl_args,
        inv_state_args,
        inv_ctrl_args,
        color=None,
        alpha=0.2,
        zorder=-10,
    ):
        if color is None:
            color = (0.5, 0.5, 0.5)
        state_traj = np.nan * np.ones((num_timesteps + 1, self._init_state.size))
        state_traj[0, :] = self._init_state.flatten()
        for kk, tt in enumerate(time_vec[:-1]):
            u = (
                self.feedback_gain[kk] @ state_traj[kk, :].reshape((-1, 1))
                + self.feedthrough_gain[kk]
            )
            state_traj[kk + 1, :] = self.prop_state(
                tt,
                state_traj[kk, :].reshape((-1, 1)),
                u,
                state_args,
                ctrl_args,
                True,
                inv_state_args,
                inv_ctrl_args,
            ).ravel()

        # plot backward pass trajectory
        extra_args = dict(
            color=color,
            alpha=alpha,
            zorder=zorder,
        )
        if len(plt_inds) == 3:
            fig.axes[self._ax].plot(
                state_traj[:, plt_inds[0]],
                state_traj[:, plt_inds[1]],
                state_traj[:, plt_inds[2]],
                **extra_args,
            )
        else:
            fig.axes[self._ax].plot(
                state_traj[:, plt_inds[0]],
                state_traj[:, plt_inds[1]],
                **extra_args,
            )

        if matplotlib.get_backend() != "Agg":
            plt.pause(0.005)
        if save_animation:
            with io.BytesIO() as buff:
                fig.savefig(buff, format="raw")
                buff.seek(0)
                arr = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                    (*(fig.canvas.get_width_height()[1::-1]), -1)
                )
                if arr.shape[0] != fig_w or arr.shape[1] != fig_h:
                    img = Image.fromarray(arr).resize((fig_w, fig_h), Image.BICUBIC)
                else:
                    img = Image.fromarray(arr)
            return img

        return []

    def reset(self, tt, cur_state, end_state):
        """Reset values for calculating the control parameters.

        This generally does not need to be called outside of the class. It is
        automatically called when calculating the control parameters.

        Parameters
        ----------
        tt : float
            Current time when starting the control calculation.
        cur_state : N x 1 numpy array
            Current state.
        end_state : N x 1 numpy array
            Desired/reference ending state.

        Returns
        -------
        old_cost : float
            prior cost.
        num_timesteps : int
            number of timesteps in the horizon.
        traj : Nh x N numpy array
            state trajectory, each row is a timestep.
        time_vec : Nh+1 numpy array
            time vector.
        """
        old_cost = float("inf")
        num_timesteps = int(self.time_horizon / self.dt)
        self._init_state = cur_state.reshape((-1, 1)).copy()
        self.end_state = end_state.reshape((-1, 1)).copy()
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

        time_vec = tt + abs_dt * np.linspace(0, 1, num_timesteps + 1, endpoint=True)

        return old_cost, num_timesteps, traj, time_vec

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
        ax_num=0,
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
            drawn, and the title is updated if supplied; the rest of the figure
            remains unchanged. The default is None which creates a new figure
            and adds a title. If no title is supplied a default is used.
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
        self._ax = ax_num

        old_cost, num_timesteps, traj, time_vec = self.reset(tt, cur_state, end_state)

        frame_list = []
        if show_animation:
            if fig is None:
                fig = plt.figure()
                if len(plt_inds) == 3:
                    extra_args = {"projection": "3d"}
                else:
                    extra_args = {}
                fig.add_subplot(1, 1, 1, **extra_args)
                try:
                    fig.axes[self._ax].set_aspect("equal", adjustable="box")
                except Exception:
                    pass

                if plt_opts is None:
                    plt_opts = gplot.init_plotting_opts(f_hndl=fig)

                if ttl is None:
                    ttl = "ELQR"

                gplot.set_title_label(
                    fig, self._ax, plt_opts, ttl=ttl, use_local=(self._ax != 0)
                )

                # draw start
                fig.axes[self._ax].scatter(
                    *self._init_state.ravel()[plt_inds],
                    marker="o",
                    color="g",
                    zorder=1000,
                )
                fig.tight_layout()
            else:
                if ttl is not None:
                    gplot.set_title_label(
                        fig, self._ax, plt_opts, ttl=ttl, use_local=(self._ax != 0)
                    )

            fig.axes[self._ax].scatter(
                *self.end_state.ravel()[plt_inds],
                marker="x",
                color="r",
                zorder=1000,
            )
            if matplotlib.get_backend() != "Agg":
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
                    arr = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                        (*(fig.canvas.get_width_height()[1::-1]), -1)
                    )
                    if arr.shape[0] != fig_w or arr.shape[1] != fig_h:
                        img = Image.fromarray(arr).resize((fig_w, fig_h), Image.BICUBIC)
                    else:
                        img = Image.fromarray(arr)
                frame_list.append(img)

        if disp:
            print("Starting ELQR optimization loop...")

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

            # quadratize final cost
            traj = self.quadratize_final_cost(
                ii, num_timesteps, traj, time_vec, cost_args
            )

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

            # get cost
            cost = 0
            x = traj[0, :].copy().reshape((-1, 1))
            for kk, tt in enumerate(time_vec[:-1]):
                u = self.feedback_gain[kk] @ x + self.feedthrough_gain[kk]
                if self.control_constraints is not None:
                    u = self.control_constraints(tt, u)
                cost += self.cost_function(
                    tt,
                    x,
                    u,
                    cost_args,
                    is_initial=(kk == 0),
                    is_final=False,
                )
                x = self.prop_state(
                    tt, x, u, state_args, ctrl_args, True, inv_state_args, inv_ctrl_args
                )
            cost += self.cost_function(
                time_vec[-1],
                x,
                u,
                cost_args,
                is_initial=False,
                is_final=True,
            )

            if show_animation:
                img = self.draw_traj(
                    fig,
                    plt_inds,
                    fig_h,
                    fig_w,
                    (save_animation and ii % self.gif_frame_skip == 0),
                    num_timesteps,
                    time_vec,
                    state_args,
                    ctrl_args,
                    inv_state_args,
                    inv_ctrl_args,
                )
                if save_animation and ii % self.gif_frame_skip == 0:
                    frame_list.append(img)

            if disp:
                print("\tIteration: {:3d} Cost: {:10.4f}".format(ii, cost))

            # check for convergence
            if np.abs((old_cost - cost) / cost) < self.tol:
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
            if self.control_constraints is not None:
                ctrl_signal[kk, :] = self.control_constraints(
                    tt, ctrl_signal[kk, :].reshape((-1, 1))
                ).ravel()

            cost += self.cost_function(
                tt,
                state_traj[kk, :].reshape((-1, 1)),
                ctrl_signal[kk, :].reshape((-1, 1)),
                cost_args,
                is_initial=(kk == 0),
                is_final=False,
            )
            state_traj[kk + 1, :] = self.prop_state(
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
            extra_args = dict(
                linestyle="-",
                color="g",
            )
            if len(plt_inds) == 3:
                fig.axes[self._ax].plot(
                    state_traj[:, plt_inds[0]],
                    state_traj[:, plt_inds[1]],
                    state_traj[:, plt_inds[2]],
                    **extra_args,
                )
            else:
                fig.axes[self._ax].plot(
                    state_traj[:, plt_inds[0]], state_traj[:, plt_inds[1]], **extra_args
                )
            if matplotlib.get_backend() != "Agg":
                plt.pause(0.001)
            if save_animation:
                with io.BytesIO() as buff:
                    fig.savefig(buff, format="raw")
                    buff.seek(0)
                    arr = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                        (*(fig.canvas.get_width_height()[1::-1]), -1)
                    )
                    if arr.shape[0] != fig_w or arr.shape[1] != fig_h:
                        img = Image.fromarray(arr).resize((fig_w, fig_h), Image.BICUBIC)
                    else:
                        img = Image.fromarray(arr)
                frame_list.append(img)

        u = ctrl_signal[0, :].reshape((-1, 1))
        details = (cost, state_traj, ctrl_signal, fig, frame_list)
        return (u, *details) if provide_details else u
