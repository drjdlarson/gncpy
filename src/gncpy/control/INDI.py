import numpy as np
import scipy.linalg as la

import gncpy.dynamics.basic as gdyn


class INDI:
    """Incremental Nonlinear Dynamic Inversion (INDI) controller.

    Implements the INDI control law:
        u = u_0 + B_0^+ @ (-x_dot_0 + r_dot + K @ e)

    where:
        u_0: current control input
        B_0: control effectiveness matrix (can be from dynamics or set directly)
        x_dot_0: current state derivative (measured or estimated)
        r_dot: reference derivative
        e = r - y: tracking error (y = H @ x, or y = x if H = I)
        K: feedback gain matrix

    For rate control (H = I):
        e = r - x (reference minus state)

    Attributes
    ----------
    omit_A_flag : bool
        If True, A matrix is not used (default for INDI). Set to False to enable A matrix (not yet implemented).
    dynObj : dynamics object
        Optional dynamics object for computing B0 online. Not needed if B0 is provided directly.
    _K : numpy array
        Feedback gain matrix.
    _H : numpy array
        Output/observation matrix. If None, assumes full state feedback (H = I).
    _B0 : numpy array
        Control effectiveness matrix. Can be set via set_state_model() or provided per control step.
    _B0_inv : numpy array
        Cached inverse/pseudo-inverse of B0 for efficiency.
    """

    def __init__(self, omit_A=True):

        super().__init__()

        self.omit_A_flag = omit_A

        self.dynObj = None
        self._dt = None

        self._K = None
        self._H = None
        self._H_inv = None
        self._B0 = None  # Control effectiveness matrix
        self._B0_inv = None  # Cached inverse/pseudo-inverse of B0

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
    def K(self):
        """Read only K matrix."""
        return self._K

    @property
    def B0(self):
        """Read only B0 (control effectiveness) matrix."""
        return self._B0

    def set_state_model(self, dynObj=None, dt=None, K=None, H=None, B0=None):
        """Set the state model and controller parameters.

        Parameters
        ----------
        dynObj : dynamics object, optional
            Dynamics object for online B0 computation. Not needed if B0 is provided.
        dt : float, optional
            Timestep.
        K : numpy array
            Feedback gain matrix.
        H : numpy array, optional
            Output matrix jacobian. If None, assumes H = I (full state feedback).
            assumes when gven H that is the linearized output matrix, this could also be C in the state space representation.
            The inverse is computed and cached for efficiency.
        B0 : numpy array, optional
            Control effectiveness matrix. If provided, its inverse is cached for efficiency.
            Can also be provided per control step if it varies over time.
        """
        self.dynObj = dynObj
        if dt is not None:
            self.dt = dt

        self._K = K
        if H is not None:
            self.update_H(H)

        # Cache B0 and its inverse if provided
        if B0 is not None:
            self.update_B0(B0)

    def update_B0(self, B0):
        """Update the cached control effectiveness matrix B0 and its inverse.

        Use this method when B0 changes but you want to cache it for multiple
        control steps without recomputing the inverse each time.

        Parameters
        ----------
        B0 : numpy array
            New control effectiveness matrix.
        """
        self._B0 = B0
        self._B0_inv = self._compute_inverse(B0)

    # TODO: the H stuff is not currently tested
    def update_H(self, H):
        """Update the output matrix H.

        Parameters
        ----------
        H : numpy array
            New output/observation matrix.
        """
        self._H = H
        self._H_inv = self._compute_inverse(H)

    def get_B0_from_dynamics(self, tt, x_hat, u_hat, state_args, ctrl_args):
        """Get control effectiveness matrix B0 from dynamics object.

        Parameters
        ----------
        tt : float
            Current time.
        x_hat : numpy array
            Current state estimate.
        u_hat : numpy array
            Current control input.
        state_args : tuple
            Additional state arguments.
        ctrl_args : tuple
            Additional control arguments.

        Returns
        -------
        B0 : numpy array
            Control effectiveness matrix.
        """
        if self.dynObj is None:
            raise RuntimeError("No dynamics object set. Provide B0 directly.")

        if self.omit_A_flag:
            # INDI doesn't use A matrix
            if isinstance(self.dynObj, gdyn.NonlinearDynamicsBase):
                if self.dynObj.dt < 0:
                    self.dynObj.dt *= -1
                B0 = self.dynObj.get_input_mat(
                    tt, x_hat, u_hat, state_args=state_args, ctrl_args=ctrl_args
                )
            else:
                B0 = self.dynObj.get_input_mat(tt, *ctrl_args)
        else:
            raise NotImplementedError("A matrix usage not yet implemented for INDI.")

        return B0

    def calculate_control(
        self,
        cur_time,
        cur_state,
        cur_state_dot,
        cur_input,
        ref,
        ref_dot,
        state_args=None,
        ctrl_args=None,
        B0=None,
    ):
        """Calculate INDI control output.

        Implements INDI control law from equation (5):
            u = u_0 + B_0^+ @ (-x_dot_0 + (dh/dx)|_x0)^-1 @ (r_dot + K @ e(t)))

        Simplified (when H = I):
            u = u_0 + B_0^+ @ (-x_dot_0 + r_dot + K @ e)

        where:
            u_0: current control input
            B_0^+: pseudo-inverse of control effectiveness matrix
            x_dot_0: current state derivative (measured/estimated)
            r_dot: reference trajectory derivative
            e = r - y: tracking error
            K: feedback gain matrix

        Parameters
        ----------
        cur_time : float
            Current time.
        cur_state : numpy array (n,)
            Current state x.
        cur_state_dot : numpy array (n,)
            Current state derivative x_dot (measured or estimated).
        cur_input : numpy array (m,)
            Current control input u_0.
        ref : numpy array (p,)
            Reference trajectory r (or r(t)).
        ref_dot : numpy array (p,)
            Reference derivative r_dot.
        state_args : tuple, optional
            Additional arguments for dynamics object.
        ctrl_args : tuple, optional
            Additional arguments for dynamics object.
        B0 : numpy array (n, m), optional
            Control effectiveness matrix for this time step. If provided, overrides
            the cached B0. Use this for time-varying B0 computed on-the-fly.

        Returns
        -------
        u : numpy array (m,)
            New control input.
        """
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()

        # Determine control effectiveness matrix B0 to use
        if B0 is not None:
            # User provided B0 for this time step (e.g., time-varying linearization)
            B0_inv = self._compute_inverse(B0)
        elif self.dynObj is not None:
            # Compute B0 from dynamics object
            B0 = self.get_B0_from_dynamics(
                cur_time, cur_state, cur_input, state_args, ctrl_args
            )
            B0_inv = self._compute_inverse(B0)
        elif self._B0 is not None:
            # Use cached B0 and its pre-computed inverse
            B0_inv = self._B0_inv
        else:
            raise RuntimeError(
                "No B0 available. Provide dynObj, set B0 via set_state_model/update_B0, or pass B0 to calculate_control."
            )

        # Compute output y = H @ x (or y = x if H = I)
        if self._H is not None:
            y = self._H @ cur_state
            H_inv = self._H_inv

        else:
            y = cur_state
            H = np.eye(len(cur_state))
            H_inv = np.eye(len(cur_state))

        # Tracking error: e = r - y
        e = ref - y

        # INDI control law: u = u_0 + B_0^+ @ (-x_dot_0 + r_dot + K @ e)

        # TODO: right now this control class need the real internal state x
        # in reality if H is not the identity so x is not equal to y then
        # we likely do not have direct access to x either, thus this class
        # need to handle when it is given y and not x, the hard part being
        # id we also want to use the dynamics object to get B0 as that object
        # the internal state.
        delta_x_dot = -cur_state_dot + H_inv @ (ref_dot + self._K @ e)
        u = cur_input + B0_inv @ delta_x_dot

        return u

    def _compute_inverse(self, mat):
        """Compute inverse or pseudo-inverse of a matrix.

        Parameters
        ----------
        mat : numpy array
            Matrix to invert.

        Returns
        -------
        numpy array
            Inverse (if square and full rank) or Moore-Penrose pseudo-inverse.
        """
        if mat.shape[0] == mat.shape[1]:
            return la.inv(mat)
        else:
            return la.pinv(mat)
