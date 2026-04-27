"""Flexible nonlinear 6DOF fixed-wing aircraft model."""

import os
import pathlib
from dataclasses import dataclass, field

import numpy as np
from ruamel.yaml import YAML

import gncpy.math as gmath
from gncpy.dynamics.basic.nonlinear_dynamics_base import NonlinearDynamicsBase


yaml = YAML()


@dataclass
class FixedWingMassParams:
    """Mass properties for a fixed-wing aircraft."""

    mass_kg: float = 1.0
    inertia_kgm2: list = field(
        default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )


@dataclass
class FixedWingAeroParams:
    """Aerodynamic coefficients.

    Coefficients that are omitted from YAML default to zero, except the drag
    model's Oswald efficiency factor, which defaults to one.
    """

    coeffs: dict = field(default_factory=dict)

    def get(self, name, default=0.0):
        """Return an aerodynamic coefficient with a stable default."""
        return self.coeffs.get(name, default)


@dataclass
class FixedWingGeometryParams:
    """Reference geometry for aerodynamic dimensionalization."""

    wing_area_m2: float = 1.0
    wing_span_m: float = 1.0
    mean_chord_m: float = 1.0
    aero_center_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    apply_aero_center_moment: bool = False


@dataclass
class FixedWingPropulsionParams:
    """Propulsion model parameters.

    When no propulsion coefficients are supplied, throttle produces no thrust.
    """

    pos_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    thrust_axis: list = field(default_factory=lambda: [1.0, 0.0, 0.0])
    poly_ct: list = field(default_factory=list)
    poly_cp: list = field(default_factory=list)
    poly_power: list = field(default_factory=list)
    prop_radius_m: float = 0.0
    omega_max_rad_s: float = 0.0
    omega_min_rad_s: float = 0.0
    power_min_w: float = 0.0
    power_max_w: float = np.inf
    include_reaction_torque: bool = False
    rotation_dir: float = 1.0


@dataclass
class FixedWingEnvironmentParams:
    """Simple environment parameters used by the dynamics model."""

    density_kgpm3: float = 1.225
    gravity_mps2: float = 9.80665
    wind_ned_mps: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class FixedWing6DOFParams:
    """Container for fixed-wing model parameters."""

    mass: FixedWingMassParams = field(default_factory=FixedWingMassParams)
    aero: FixedWingAeroParams = field(default_factory=FixedWingAeroParams)
    geometry: FixedWingGeometryParams = field(default_factory=FixedWingGeometryParams)
    propulsion: FixedWingPropulsionParams = field(
        default_factory=FixedWingPropulsionParams
    )
    environment: FixedWingEnvironmentParams = field(default_factory=FixedWingEnvironmentParams)


def _merge_dataclass(obj, data):
    """Apply mapping values to a dataclass instance."""
    if data is None:
        return obj
    for key, val in dict(data).items():
        if hasattr(obj, key):
            setattr(obj, key, val)
    return obj


def _load_params(params):
    """Load fixed-wing parameters from None, a mapping, or a YAML file."""
    out = FixedWing6DOFParams()
    if params is None:
        return out

    if isinstance(params, (str, os.PathLike)):
        with open(params, "r") as fin:
            params = yaml.load(fin)

    params = dict(params)
    _merge_dataclass(out.mass, params.get("mass"))
    _merge_dataclass(out.geometry, params.get("geometry"))
    _merge_dataclass(out.propulsion, params.get("propulsion"))
    _merge_dataclass(out.environment, params.get("environment"))

    aero = params.get("aero")
    if aero is not None:
        if "coeffs" in aero:
            out.aero.coeffs.update(dict(aero["coeffs"]))
        else:
            out.aero.coeffs.update(dict(aero))
    return out


class fw_smap:
    """Indices for the compact fixed-wing state vector."""

    ned_pos = [0, 1, 2]
    body_vel = [3, 4, 5]
    quat = [6, 7, 8, 9]
    body_rot_rate = [10, 11, 12]


class FixedWing6DOF(NonlinearDynamicsBase):
    r"""Nonlinear fixed-wing 6DOF model with quaternion attitude.

    The state is ``[pn, pe, pd, u, v, w, qw, qx, qy, qz, p, q, r]``.
    The control input is ``[delta_e, delta_a, delta_r, throttle]`` with an
    optional fifth flap input ``delta_f``. Missing aerodynamic coefficients
    default to zero so low-fidelity YAML files can provide only the terms they
    need.
    """

    state_names = (
        "north pos",
        "east pos",
        "down pos",
        "body vel x",
        "body vel y",
        "body vel z",
        "quat qw",
        "quat qx",
        "quat qy",
        "quat qz",
        "body rate p",
        "body rate q",
        "body rate r",
    )
    control_names = ("elevator", "aileron", "rudder", "throttle", "flap")
    state_map = fw_smap

    def __init__(self, params=None, params_file=None, library_dir=None, **kwargs):
        super().__init__(**kwargs)
        if params is not None and params_file is not None:
            raise ValueError("Specify either params or params_file, not both")
        if params_file is not None:
            params = self.validate_params_file(params_file, library_dir=library_dir)
        self.params = _load_params(params)

    @property
    def cont_fnc_lst(self):
        """Continuous functions used by the base numerical Jacobian path."""
        return [
            lambda t, x, *args, ii=ii: self.state_derivative(
                t, x, None if len(args) == 0 else args[0]
            )[ii]
            for ii in range(len(self.state_names))
        ]

    def validate_params_file(self, params_file, library_dir=None):
        """Return the resolved path to a YAML parameter file."""
        has_sep = os.path.sep in params_file
        if os.path.altsep is not None:
            has_sep = has_sep or os.path.altsep in params_file

        if os.path.isabs(params_file) or has_sep:
            cf = params_file
        else:
            cf = os.path.join(os.getcwd(), params_file)
            if not os.path.isfile(cf):
                base_dir = (
                    pathlib.Path(__file__).parent.resolve()
                    if library_dir is None
                    else pathlib.Path(library_dir)
                )
                cf = os.path.join(base_dir, params_file)
        if not os.path.isfile(cf):
            raise FileNotFoundError("Failed to find config file {}".format(params_file))
        return cf

    def _parse_control(self, u):
        if u is None:
            u = np.zeros(4)
        u = np.asarray(u, dtype=float).ravel()
        if u.size not in (4, 5):
            raise ValueError("Control input must have 4 or 5 elements")
        delta_e, delta_a, delta_r, throttle = u[:4]
        delta_f = 0.0 if u.size == 4 else u[4]
        return delta_e, delta_a, delta_r, np.clip(throttle, 0.0, 1.0), delta_f

    def _aero_angles(self, body_vel):
        airspeed = np.linalg.norm(body_vel)
        if airspeed <= np.finfo(float).eps:
            return 0.0, 0.0, 0.0
        alpha = np.arctan2(body_vel[2], body_vel[0])
        beta = np.arcsin(np.clip(body_vel[1] / airspeed, -1.0, 1.0))
        return airspeed, alpha, beta

    def _resolve_wind_ned(self, state_args=None):
        wind_ned = np.asarray(self.params.environment.wind_ned_mps, dtype=float)
        if wind_ned.shape != (3,):
            raise ValueError("Environment wind_ned_mps must be a 3-vector")

        if state_args is None:
            return wind_ned
        if isinstance(state_args, tuple):
            if len(state_args) == 0:
                return wind_ned
            if len(state_args) != 1:
                raise ValueError("state_args for FixedWing6DOF must be empty or (wind_arg,)")
            state_args = state_args[0]

        if isinstance(state_args, dict):
            wind_arg = state_args.get("wind_ned", wind_ned)
        else:
            wind_arg = state_args

        wind_ned = np.asarray(wind_arg, dtype=float).ravel()
        if wind_ned.shape != (3,):
            raise ValueError("Wind argument must be a 3-vector in NED coordinates")
        return wind_ned

    def calc_air_data(self, state, state_args=None):
        """Return true air-relative quantities for the current state.

        Wind is treated as an exogenous disturbance. It may be supplied either
        via ``state_args=(wind_ned,)`` or ``state_args=({"wind_ned": wind_ned},)``.
        When omitted, :attr:`FixedWingEnvironmentParams.wind_ned_mps` is used.
        """
        x = np.asarray(state, dtype=float).ravel()
        quat = gmath.quat_normalize(x[fw_smap.quat])
        body_vel = x[fw_smap.body_vel]
        wind_ned = self._resolve_wind_ned(state_args)
        wind_body = gmath.quat_rotate_vector(gmath.quat_conjugate(quat), wind_ned)
        air_vel_body = body_vel - wind_body
        airspeed, alpha, beta = self._aero_angles(air_vel_body)
        return {
            "wind_ned": wind_ned,
            "wind_body": wind_body,
            "air_vel_body": air_vel_body,
            "airspeed": airspeed,
            "alpha": alpha,
            "beta": beta,
        }

    def _nondim_rates(self, body_rates, airspeed):
        if airspeed <= np.finfo(float).eps:
            return 0.0, 0.0, 0.0
        p, q, r = body_rates
        geom = self.params.geometry
        return (
            p * geom.wing_span_m / (2 * airspeed),
            q * geom.mean_chord_m / (2 * airspeed),
            r * geom.wing_span_m / (2 * airspeed),
        )

    def _calc_aero_force_mom(self, state, u, state_args=None):
        body_rates = state[fw_smap.body_rot_rate]
        delta_e, delta_a, delta_r, _, delta_f = self._parse_control(u)

        air_data = self.calc_air_data(state, state_args=state_args)
        airspeed = air_data["airspeed"]
        alpha = air_data["alpha"]
        beta = air_data["beta"]
        rho = self.params.environment.density_kgpm3
        qbar = 0.5 * rho * airspeed**2
        geom = self.params.geometry
        aero = self.params.aero

        p_hat, q_hat, r_hat = self._nondim_rates(body_rates, airspeed)
        cl = (
            aero.get("CL0")
            + aero.get("CL_alpha") * alpha
            + aero.get("CL_q") * q_hat
            + aero.get("CL_delta_e") * delta_e
            + aero.get("CL_delta_a") * delta_a
            + aero.get("CL_delta_r") * delta_r
            + aero.get("CL_delta_f") * delta_f
        )
        aspect_ratio = geom.wing_span_m**2 / geom.wing_area_m2
        cd = (
            aero.get("CD0", aero.get("CD_min"))
            + (cl - aero.get("CL_minD")) ** 2
            / (np.pi * aero.get("oswald_eff", 1.0) * aspect_ratio)
            + aero.get("CD_alpha") * alpha
            + aero.get("CD_delta_e") * delta_e
            + aero.get("CD_delta_a") * delta_a
            + aero.get("CD_delta_r") * delta_r
            + aero.get("CD_delta_f") * delta_f
        )
        cy = (
            aero.get("CY_beta") * beta
            + aero.get("CY_p") * p_hat
            + aero.get("CY_r") * r_hat
            + aero.get("CY_delta_a") * delta_a
            + aero.get("CY_delta_r") * delta_r
            + aero.get("CY_delta_f") * delta_f
        )
        c_roll = (
            aero.get("Cl_beta") * beta
            + aero.get("Cl_p") * p_hat
            + aero.get("Cl_r") * r_hat
            + aero.get("Cl_delta_a") * delta_a
            + aero.get("Cl_delta_r") * delta_r
            + aero.get("Cl_delta_f") * delta_f
        )
        cm = (
            aero.get("Cm0")
            + aero.get("Cm_alpha") * alpha
            + aero.get("Cm_q") * q_hat
            + aero.get("Cm_delta_e") * delta_e
            + aero.get("Cm_delta_a") * delta_a
            + aero.get("Cm_delta_f") * delta_f
        )
        cn = (
            aero.get("Cn_beta") * beta
            + aero.get("Cn_p") * p_hat
            + aero.get("Cn_r") * r_hat
            + aero.get("Cn_delta_a") * delta_a
            + aero.get("Cn_delta_r") * delta_r
            + aero.get("Cn_delta_f") * delta_f
        )

        lift = qbar * geom.wing_area_m2 * cl
        drag = qbar * geom.wing_area_m2 * cd
        side = qbar * geom.wing_area_m2 * cy
        f_wind = np.array([-drag, side, -lift])
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)
        c_beta = np.cos(beta)
        s_beta = np.sin(beta)
        r_y_neg_alpha = np.array(
            [[c_alpha, 0.0, -s_alpha], [0.0, 1.0, 0.0], [s_alpha, 0.0, c_alpha]]
        )
        r_z_beta = np.array(
            [[c_beta, -s_beta, 0.0], [s_beta, c_beta, 0.0], [0.0, 0.0, 1.0]]
        )
        force = r_y_neg_alpha @ r_z_beta @ f_wind

        moment = np.array(
            [
                qbar * geom.wing_area_m2 * geom.wing_span_m * c_roll,
                qbar * geom.wing_area_m2 * geom.mean_chord_m * cm,
                qbar * geom.wing_area_m2 * geom.wing_span_m * cn,
            ]
        )
        if geom.apply_aero_center_moment:
            r_ac = np.asarray(geom.aero_center_m, dtype=float)
            moment += np.cross(r_ac, force)
        return force, moment

    def _calc_prop_force_mom(self, state, u, state_args=None):
        _, _, _, throttle, _ = self._parse_control(u)
        prop = self.params.propulsion
        axis = np.asarray(prop.thrust_axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= np.finfo(float).eps:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / axis_norm

        if not prop.poly_ct or prop.prop_radius_m <= 0 or prop.omega_max_rad_s <= 0:
            return np.zeros(3), np.zeros(3)

        airspeed = self.calc_air_data(state, state_args=state_args)["airspeed"]
        rho = self.params.environment.density_kgpm3
        diam = 2 * prop.prop_radius_m
        omega = np.clip(
            throttle * prop.omega_max_rad_s, prop.omega_min_rad_s, prop.omega_max_rad_s
        )
        n_rev = omega / (2 * np.pi)
        advance_ratio = (
            0.0 if n_rev <= np.finfo(float).eps else airspeed / (n_rev * diam)
        )
        ct = np.polynomial.Polynomial(prop.poly_ct[-1::-1])(advance_ratio)
        thrust = ct * rho * n_rev**2 * diam**4
        force = thrust * axis
        moment = np.cross(np.asarray(prop.pos_m, dtype=float), force)

        if prop.include_reaction_torque and prop.poly_cp and omega > np.finfo(float).eps:
            cp = np.polynomial.Polynomial(prop.poly_cp[-1::-1])(advance_ratio)
            prop_power = cp * rho * n_rev**3 * diam**5
            torque = prop_power / omega
            moment += -prop.rotation_dir * torque * axis
        return force, moment

    def _calc_grav_force(self, quat):
        gravity_ned = np.array([0.0, 0.0, self.params.environment.gravity_mps2])
        force_ned = self.params.mass.mass_kg * gravity_ned
        return gmath.quat_rotate_vector(gmath.quat_conjugate(quat), force_ned)

    def calc_force_mom(self, state, u=None, state_args=None):
        """Calculate total body-frame force and moment."""
        x = np.asarray(state, dtype=float).ravel()
        quat = gmath.quat_normalize(x[fw_smap.quat])
        aero_force, aero_moment = self._calc_aero_force_mom(x, u, state_args=state_args)
        prop_force, prop_moment = self._calc_prop_force_mom(x, u, state_args=state_args)
        grav_force = self._calc_grav_force(quat)
        return aero_force + prop_force + grav_force, aero_moment + prop_moment

    def state_derivative(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        """Return the continuous-time state derivative."""
        x = np.asarray(state, dtype=float).ravel()
        quat = gmath.quat_normalize(x[fw_smap.quat])
        body_vel = x[fw_smap.body_vel]
        omega = x[fw_smap.body_rot_rate]
        force, moment = self.calc_force_mom(x, u, state_args=state_args)

        xdot = np.zeros(len(self.state_names))
        xdot[fw_smap.ned_pos] = gmath.quat_rotate_vector(quat, body_vel)
        xdot[fw_smap.body_vel] = force / self.params.mass.mass_kg - np.cross(
            omega, body_vel
        )

        qw, qx, qy, qz = quat
        p, q, r = omega
        xdot[fw_smap.quat] = 0.5 * np.array(
            [
                -p * qx - q * qy - r * qz,
                p * qw + r * qy - q * qz,
                q * qw - r * qx + p * qz,
                r * qw + q * qx - p * qy,
            ]
        )

        inertia = np.asarray(self.params.mass.inertia_kgm2, dtype=float)
        xdot[fw_smap.body_rot_rate] = np.linalg.solve(
            inertia, moment - np.cross(omega, inertia @ omega)
        )
        return xdot.reshape((-1, 1))

    def _cont_dyn(self, t, x, u, state_args, ctrl_args):
        return self.state_derivative(t, x, u, state_args=state_args, ctrl_args=ctrl_args)

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        next_state = super().propagate_state(
            timestep, state, u=u, state_args=state_args, ctrl_args=ctrl_args
        )
        next_state[fw_smap.quat] = gmath.quat_normalize(
            next_state[fw_smap.quat].ravel()
        ).reshape((4, 1))
        return next_state

    def get_continuous_state_mat(self, timestep, state, u=None, state_args=None):
        """Return the continuous-time state Jacobian.

        Wind supplied through ``state_args`` is held fixed during linearization.
        """
        trim_u = np.zeros((4, 1)) if u is None else u
        A, _ = gmath.linearize_dynamics(
            lambda x, uu, t: self.state_derivative(
                t, x, uu, state_args=state_args
            ).ravel(),
            state,
            trim_u,
        )
        return A

    def get_continuous_input_mat(self, timestep, state, u, state_args=None):
        """Return the continuous-time input Jacobian.

        Wind supplied through ``state_args`` is held fixed during linearization.
        """
        _, B = gmath.linearize_dynamics(
            lambda x, uu, t: self.state_derivative(
                t, x, uu, state_args=state_args
            ).ravel(),
            state,
            u,
        )
        return B

    def get_state_mat(
        self, timestep, state, *f_args, u=None, ctrl_args=None, use_continuous=False
    ):
        if use_continuous:
            return self.get_continuous_state_mat(timestep, state, u=u, state_args=f_args)
        return super().get_state_mat(
            timestep,
            state,
            *f_args,
            u=u,
            ctrl_args=ctrl_args,
            use_continuous=False,
        )

    def get_input_mat(
        self, timestep, state, u, state_args=None, ctrl_args=None, use_continuous=False
    ):
        """Return the input Jacobian.

        By default this returns the discrete-time Jacobian of
        :meth:`propagate_state`, matching :class:`NonlinearDynamicsBase`.
        """
        if use_continuous:
            return self.get_continuous_input_mat(
                timestep, state, u, state_args=state_args
            )
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        _, B = gmath.linearize_dynamics(
            lambda x, uu, t: self.propagate_state(
                t,
                x.reshape((-1, 1)),
                u=uu.reshape((-1, 1)),
                state_args=state_args,
                ctrl_args=ctrl_args,
            ).ravel(),
            state,
            u,
        )
        return B
