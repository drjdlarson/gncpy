import numpy as np
from .nonlinear_dynamics_base import NonlinearDynamicsBase
import pyatmos as atm
import gncpy.wgs84 as wgs84
import copy

class ReentryVehicle(NonlinearDynamicsBase):
    """general model for a ballistic reentry vehicle

    All dynamics done in ENU Earth frame (noninertial frame)
    Control model is ENU acceleration

    Notes
    -----
    This is taken from:
        Survey of Maneuvering
        Target Tracking.
        Part II: Motion Models of
        Ballistic and Space Targets
        X. RONG LI, Fellow, IEEE
        VESSELIN P. JILKOV, Member, IEEE

    """

    # all states in m or m/s
    state_names = (
        "E pos",
        "N pos",
        "U pos",
        "E vel",
        "N vel",
        "U vel",
    )

    def __init__(
        self,
        dt=0.01,
        ballistic_coefficient=5000,
        origin_lat=-77.0563,
        origin_long=38.8719,
        CD0 = 0.25,
        has_thrusters = True,
        atm_model = atm.expo,
        **kwargs,
    ):
        """Initialize an object.

        Parameters
        ----------
        dt : float, optional
            time period for integration
        ballistic_coefficient : float, optional
            ballistic coefficient, = m/(S*CD) (kg/m2)
        origin_lat: float, optional
            Latitude of coordinate frame origin, used for calculating noninertial accelerations due to Earth rotation
            Defaults to the Pentagon
        origin_long: float, optional
            Longitude of coordinate frame origin, used for calculating noninertial accelerations due to Earth rotation
            Defaults to the Pentagon
        CD0: float, optional
            zero-lift drag coefficient; only needed for determining induced drag for manuevering RV's. 
            If control input is None, this parameter affects nothing and is inconsequential. 
        has_thrusters: bool, optional
            Determines if the vehicle will respond to commanded turn/climb accelerations beyond what it can do with aero
            forces alone (typically applies in upper/exo-atmosphere). If false, control commands will be limited to 
            accelerations possible from aero forces alone. 
        atm_model: pyatmos class, optional
            determines how density will be calculated for different vehicle altitudes. Either atm.expo or atm.coesa76 
            can be chosen. Expo runs ~10x faster thatn coesa76. If coesa76 chosen, it currently takes up 50% of runtime for
            a model with control input. Thus, choosing expo will ~halve overall runtime. Exponential model underestimates 
            atmospheric density by up to 25%. 
        **kwargs : dict
            Additional arguments for the parent class.
        """
        super().__init__(dt=dt, **kwargs)
        self.drag_parameter = 1 / ballistic_coefficient
        self.origin_lat = origin_lat
        self.origin_long = origin_long
        self.CD0 = CD0
        self.has_thrusters = has_thrusters
        self.atm_model = atm_model

        '''
        control model is specific forces (in m/s2) in the velocity-turn-climb frame:
        u = np.array([a_thrust, # in direction of vehicle velocity
                      a_turn,   # in direction (left turn positive) perpendicular to velocity in the horizontal ENU plane
                      a_climb   # in direction (up positive) perpendicular to velocity and turn
                    ])
        '''
        def g0(t, x, u, *args):
            return 0

        def g1(t, x, u, *args):
            return 0

        def g2(t, x, u, *args):
            return 0

        def ENU_control_acceleration(t, x, u, *args):
            # FIXME this function assumes turning is done by aero forces only up to CL=3, then assumes any further 
            # commanded turning/climbing is done via thrusters. It also assumes a CL of 3 is achievable to account
            # for the thrust vector helping turn at some alpha. It's all very sus as an approximation


            # assumes u is np.array([a_v, a_t, a_c])

            # define VTC -> ENU rotation matrix
            v = np.linalg.norm(x[3:])
            vg = np.linalg.norm(x[3:5])
            T_ENU_VTC = np.array([[x[3]/v, -x[4]/vg, -x[3]*x[5]/(v*vg)],
                                  [x[4]/v, x[3]/vg, -x[4]*x[5]/(v*vg)],
                                  [x[5]/v, 0, vg**2/(v*vg)]])

            # calculate dynamic pressure
            rho_NED = np.array([x[1], x[0], -(x[2] + wgs84.EQ_RAD)]) # spherical approximation
            veh_alt = np.linalg.norm(rho_NED) - wgs84.EQ_RAD
            #coesa76_geom = atm.expo([veh_alt / 1000])  # geometric altitudes in km
            #density = coesa76_geom.rho  # [kg/m^3]
            density = self.density # the dynamics model MUST be called before the control model for this to be defined
            q = 1/2*density*v**2

            # limit maximum lift, assume CLMax = 3
            # (set high to poorly account for fact that thrust can be unaligned with velocity to help turn)
            # If aero lift not enough to satisfy control, assume remaining maneuvering 
            # accelerations provided by thrusters
            u_limited = copy.deepcopy(u).astype(float)
            total_lift_acceleration = np.linalg.norm(u[1:])
            lift_accel_max = 3*q*self.drag_parameter/self.CD0
            
            if total_lift_acceleration > lift_accel_max:
                if self.has_thrusters:
                    total_lift_acceleration = lift_accel_max
                else:
                    u_limited[1:] = u_limited[1:]*lift_accel_max/total_lift_acceleration
                    total_lift_acceleration = np.linalg.norm(u_limited[1:])

            # calculate additional induced drag
            # following equation derived from CDL=CL**2/4 result from slender body potential flow theory (cite Cronvich Missile Aerodynamics)
            induced_drag_acceleration = total_lift_acceleration**2/self.drag_parameter*self.CD0/(4*q)
            u_limited[0] = u_limited[0] - induced_drag_acceleration # subtract induced drag from thrust

            # convert VTC accelerations to ENU
            a_control_ENU = T_ENU_VTC @ u_limited

            return a_control_ENU

        def g3(t, x, u, *args):
            if u is None:
                return 0
            else:
                control_accel = ENU_control_acceleration(t, x, u, *args)
                return control_accel[0]

        def g4(t, x, u, *args):
            if u is None:
                return 0
            else:
                control_accel = ENU_control_acceleration(t, x, u, *args)
                return control_accel[1]

        def g5(t, x, u, *args):
            if u is None:
                return 0
            else:
                control_accel = ENU_control_acceleration(t, x, u, *args)
                return control_accel[2]

        self._control_model = [g0, g1, g2, g3, g4, g5]

    @property
    def control_model(self):
        return self._control_model

    @control_model.setter
    def control_model(self, model):
        self._control_model = model

    @property
    def ballistic_coefficient(self):
        """Read only ballistic coefficient."""
        return 1 / self.drag_parameter

    # 3D vector acceleration function that the following functions break into components
    def ENU_acceleration(self, t, x, *args):
        # All calculations done in NED frame then converted to ENU immediately before returning

        # approximate vehicle gravity using WGS84 gravity model and spherical approximation
        dlat = x[1] / wgs84.EQ_RAD * 180 / np.pi # use (very) poor latitude approximation
        veh_lat = self.origin_lat + dlat
        rho_NED = np.array([x[1], x[0], -(x[2] + wgs84.EQ_RAD)]) # spherical approximation
        veh_alt = np.linalg.norm(rho_NED) - wgs84.EQ_RAD
        a_gravity_magnitude = wgs84.calc_gravity(veh_lat, veh_alt).T.flatten()
        a_gravity = -a_gravity_magnitude[2]*rho_NED/np.linalg.norm(rho_NED) # direct gravity towards earth center for spherical earth

        # all these calculations done in NED frame
        NED_velocity = np.array([x[4], x[3], -x[5]])
        omega_earth = wgs84.calc_earth_rate(veh_lat)
        omega_earth = omega_earth.T.flatten() # convert to row array for np.cross usage
        a_centrifugal = np.cross(omega_earth, np.cross(omega_earth, rho_NED))
        a_coriolis = 2 * np.cross(omega_earth, NED_velocity)

        a_earth_induced = a_gravity - a_centrifugal - a_coriolis

        atmosphere = self.atm_model([veh_alt / 1000])  # geometric altitudes in km by default
        density = atmosphere.rho  # [kg/m^3]
        self.density = density # store density at current altitude so we don't need to recalc for control
        velocity_norm = np.linalg.norm(NED_velocity)
        a_drag = (
            -1
            / 2
            * density
            * (velocity_norm) ** 2
            * self.drag_parameter
            * NED_velocity
            / velocity_norm
        )

        a_total_NED = a_drag + a_earth_induced
        a_total_ENU = np.copy(a_total_NED)
        a_total_ENU[0], a_total_ENU[1], a_total_ENU[2] = a_total_NED[1], a_total_NED[0], -a_total_ENU[2]

        return a_total_ENU

    @property
    def cont_fnc_lst(self):
        """Continuous time dynamics.

        Returns
        -------
        list
            functions of the form :code:`(t, x, *args)`.
        """

        # returns E velocity
        def f0(t, x, *args):
            return x[3]

        # returns N velocity
        def f1(t, x, *args):
            return x[4]

        # returns U velocity
        def f2(t, x, *args):
            return x[5]

        # returns E acceleration
        def f3(t, x, *args):
            a_total = self.ENU_acceleration(t, x, *args)
            return a_total[0]

        # returns N acceleration
        def f4(t, x, *args):
            a_total = self.ENU_acceleration(t, x, *args)
            return a_total[1]

        # returns U acceleration
        def f5(t, x, *args):
            a_total = self.ENU_acceleration(t, x, *args)
            return a_total[2]

        return [f0, f1, f2, f3, f4, f5]
    
    # overwrite continuous dynamics function to only call ENU_acceleration once
    def _cont_dyn(self, t, x, u, state_args, ctrl_args):
        r"""Implements the continuous time dynamics.

        This automatically sets up the combined differential equation based on
        the supplied continuous function list.

        This implements the equation :math:`\dot{x} = f(t, x) + g(t, x, u)` and
        returns the state derivatives as a vector. Note the control input is
        only used if an appropriate control model is set by the user.

        Parameters
        ----------
        t : float
            timestep.
        x : N x 1 numpy array
            current state.
        u : Nu x 1 numpy array
            Control effort, not used if no control model.
        state_args : tuple
            Additional arguements for the state functions.
        ctrl_args : tuple
            Additional arguments for the control functions. Not used if no
            control model

        Returns
        -------
        x_dot : N x 1 numpy array
            state derivative
        """
        out = np.zeros((len(self.state_names), 1))
        out[:3] = np.array([x[3], x[4], x[5]]).reshape(-1, 1)
        acceleration = self.ENU_acceleration(t, x)
        out[3:] = acceleration.reshape(-1, 1)
    
        if self._control_model is not None:
            for ii, g in enumerate(self._control_model):
                out[ii] += g(t, x, u, *ctrl_args)
        return out
