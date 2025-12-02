import numpy as np
import scipy.linalg as la

import gncpy.dynamics.basic as gdyn 

class INDI: 
    def __init__(self,omit_A=True):

        super().__init__()

        self.omit_A_flag = omit_A

        self.dynObj = None
        self._dt = None 

        self._K = None
        
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
    
    def set_state_model(self, dynObj=None, dt=None, K=None, H=None):
        self.dynObj = dynObj
        if dt is not None:
            self.dt = dt

        self._K = K
        self._H = H 

    def get_state_space(self, tt, x_hat, u_hat, state_args, ctrl_args): 
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

    def calculate_control(self,cur_time,cur_state,cur_state_dot,cur_input,ref,ref_dot, state_args=None, ctrl_args=None):
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = () 

        F, G = self.get_state_space(
            cur_time, cur_state, cur_input, state_args, ctrl_args
        )

        u = cur_input + self._inv_mat(G) @ (-cur_state_dot + self._inv_mat(self._H) @ (ref_dot + self._K @ (ref - self._H @ cur_state)))

        return u
    
    def _inv_mat(A):
        if A.shape[0] == A.shape[1]:
            return la.inv(A)
        else:
            return la.pinv(A)