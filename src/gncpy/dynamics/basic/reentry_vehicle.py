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
        **kwargs : dict
            Additional arguments for the parent class.
        """
        super().__init__(dt=dt, **kwargs)
        self.drag_parameter = 1 / ballistic_coefficient
        self.origin_lat = origin_lat
        self.origin_long = origin_long
        self.CD0 = CD0

        '''
        control model is accelerations (in m/s) in the velocity-turn-climb frame:
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
            # FIXME
            # this function assumes all turning is done by aerodynamic forces 
            # (the thrust vector is aligned with velocity and does not contribute to turning).
            # This assumption means no exoatmospheric turning is possible, which doesn't really work for an interceptor.

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
            coesa76_geom = atm.coesa76([veh_alt / 1000])  # geometric altitudes in km
            density = coesa76_geom.rho  # [kg/m^3]
            q = 1/2*density*v**2

            # limit maximum lift, assume CLMax = 3
            # (set high to poorly account for fact that thrust can be unaligned with velocity to help turn)
            u_limited = copy.deepcopy(u).astype(float)
            total_lift_acceleration = np.linalg.norm(u[1:])
            lift_accel_max = 3*q*self.drag_parameter/self.CD0
            if total_lift_acceleration > lift_accel_max:
                u_limited[1:] = u_limited[1:]*lift_accel_max/total_lift_acceleration
                total_lift_acceleration = np.linalg.norm(u_limited[1:])

            # calculate additional induced drag
            # following equation derived from CDL=CL**2/4 result from slender body potential flow theory (cite Cronvich Missile Aerodynamics)
            induced_drag_acceleration = total_lift_acceleration**2/self.drag_parameter*self.CD0/(4*q)
            u_limited[0] = u_limited[0] - induced_drag_acceleration # subtract induced drag from thrust

            # convert VTC accelerations to ENU
            a_control_ENU = T_ENU_VTC @ u_limited

            return a_control_ENU

        # FIXME change these to calculate induced drag from maneuvering
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

        # try to write a generic acceleration function that the following functions can break up; will this work?
        def ENU_acceleration(t, x, *args):
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

            coesa76_geom = atm.coesa76(
                [veh_alt / 1000]
            )  # geometric altitudes in km by default
            density = coesa76_geom.rho  # [kg/m^3]
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

        # returns E acceleration
        def f3(t, x, *args):
            a_total = ENU_acceleration(t, x, *args)
            return a_total[0]

        # returns N acceleration
        def f4(t, x, *args):
            a_total = ENU_acceleration(t, x, *args)
            return a_total[1]

        # returns U acceleration
        def f5(t, x, *args):
            a_total = ENU_acceleration(t, x, *args)
            return a_total[2]

        return [f0, f1, f2, f3, f4, f5]
