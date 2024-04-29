import numpy as np
import scipy.linalg as la

# FIXME need to add in a way to output cost to go's to these algorithms

class Pronav:
    r"""Implements a proportional navigation (Pronav) controller.

    Notes
    -----
    Pronav is a simplified case of ELQR for the target interception problem. 
    This class implements multiple types of pronav:
        PN: classic pronav:         assumes constant target velocity
        APN: augmented pronav:      assumes constant target acceleration
        EPN: extended pronav:       assumes constant target jerk 
        OGL: optimal guidance law:  assumes constant target acceleration and an acceleration command lag
    
    Formulation of the problem and guidance laws are taken from "modern homing missile guidance theory and techniques" from Johns Hopkins APL
    """

    def __init__(self, maneuver_time_constant=0.0):
        """Initialize an object.

        Parameters
        ----------
        maneuver_time_constant: float, optional
            Defines the time constant a_actual(s) / a_commanded(s)  = 1/(Ts+1),
            Where T is the maneuver time constant. This is only used in the OGL method. 
        """
        super().__init__() # FIXME do I need this?
        self._T = maneuver_time_constant
        

    @property
    def T(self):
        """Read maneuver time constant"""
        return self._T

    @T.setter
    def T(self, val):
        self._T = val

    def estimate_tgo(self, x_relative, v_relative):
        # where x_rel = x_target - x_ego, same for v_rel
        return -np.dot(x_relative, v_relative)/np.dot(v_relative, v_relative)

    def PN(self, x_relative, v_relative, t_go):
        r'''
        Implements pure proportional navigation. Assumes target velocity is constant and there is no 
        lag in ego vehicle acceleration commands

        Outputs commanded accelerations in the same coordinate frame that x_relative and v_relative 
        were input. 

        Note: this outputs a 3D control command. Limiting this control to axis perpendicular to vehicle
        velocity (so this doesn't command a change in thrust) must happen outside this function.
        '''

        u = 3/t_go**2 * (x_relative + v_relative * t_go)
        return u
    
    def APN(self, x_relative, v_relative, a_target, t_go):
        r'''
        Implements augmented proportional navigation. Assumes target acceleration is constant and there is no 
        lag in ego vehicle acceleration commands

        Outputs commanded accelerations in the same coordinate frame that x_relative and v_relative 
        were input. 

        Note: this outputs a 3D control command. Limiting this control to axis perpendicular to vehicle
        velocity (so this doesn't command a change in thrust) must happen outside this function.
        '''

        u = 3/t_go**2 * (x_relative + v_relative * t_go + 1/2*a_target*t_go**2)
        return u
    
    def EPN(self, x_relative, v_relative, a_target, j_target, t_go):
        r'''
        Implements extended proportional navigation. Assumes target jerk is constant and there is no 
        lag in ego vehicle acceleration commands

        Outputs commanded accelerations in the same coordinate frame that x_relative and v_relative 
        were input. 

        Note: this outputs a 3D control command. Limiting this control to axis perpendicular to vehicle
        velocity (so this doesn't command a change in thrust) must happen outside this function.
        '''

        u = 3/t_go**2 * (x_relative + v_relative * t_go + 1/2*a_target*t_go**2 + 1/6*j_target*t_go**3)
        return u
    
    def OGL(self, x_relative, v_relative, a_target, a_ego, t_go):
        r'''
        Implements the optimal guidance law for the interception problem. 
        Assumes target acceleration is constant and incorporates lag in commanded ego vehicle accelerations

        Outputs commanded accelerations in the same coordinate frame that x_relative and v_relative 
        were input. 

        Note: this outputs a 3D control command. Limiting this control to axis perpendicular to vehicle
        velocity (so this doesn't command a change in thrust) must happen outside this function.
        '''

        # split u into components to improve code readability:
        u = 6*(t_go/self.T)**2/t_go**2
        lag_term = (t_go/self.T + np.exp(-t_go/self.T) - 1)
        u *= lag_term
        u *= (x_relative + v_relative*t_go + 1/2*t_go**2*a_target - self.T**2*(lag_term)*a_ego)
        u /= (3 + 6*t_go/self.T - 6*(t_go/self.T)**2 + 2*(t_go/self.T)**3 - 12*(t_go/self.T)*np.exp(-t_go/self.T) - 3*np.exp(-2*t_go/self.T))
        return u
    
    def enu2vtc(self, x, u_ENU):
        # converts a control vector u in the enu coordinate system to the vtc coordinate system
        v = np.linalg.norm(x[3:])
        vg = np.linalg.norm(x[3:5])

        if v == 0 or vg == 0:
            print(f'v = {v}, vg = {vg}')
            raise ValueError("Velocity magnitude or ground speed magnitude is zero.")

        T_ENU_VTC = np.array([[x[3]/v, -x[4]/vg, -x[3]*x[5]/(v*vg)],
                                  [x[4]/v, x[3]/vg, -x[4]*x[5]/(v*vg)],
                                  [x[5]/v, 0, vg**2/(v*vg)]])
        u_VTC = np.linalg.inv(T_ENU_VTC) @ u_ENU
        return u_VTC











