import numpy as np
from scipy import integrate
import enum
import pathlib
import os
from ruamel.yaml import YAML
from warnings import warn
from copy import deepcopy

from gncpy.dynamics.basic import DynamicsBase
import gncpy.dynamics.aircraft.lager_super_bindings as super_bind
from gncpy.coordinate_transforms import ned_to_LLA
import gncpy.wgs84 as wgs84


yaml = YAML()
r2d = 180.0 / np.pi
d2r = 1 / r2d


class AeroParams:
    def __init__(self):
        self.cd = 0


class MassParams:
    def __init__(self):
        self.cg_m = []
        self.mass_kg = 0
        self.inertia_kgm2 = []


class PropParams:
    def __init__(self):
        self.poly_thrust = []
        self.poly_torque = []


class MotorParams:
    def __init__(self):
        self.pos_m = []
        self.dir = []

    @property
    def num_motors(self):
        return len(self.dir)


class GeoParams:
    def __init__(self):
        self.front_area_m2 = []


class AircraftParams:
    def __init__(self):
        self.aero = AeroParams()
        self.mass = MassParams()
        self.geo = GeoParams()
        self.prop = PropParams()
        self.motor = MotorParams()


yaml.register_class(AeroParams)
yaml.register_class(MassParams)
yaml.register_class(PropParams)
yaml.register_class(MotorParams)
yaml.register_class(GeoParams)
yaml.register_class(AircraftParams)


class Effector:
    def step(self, input_cmds):
        return input_cmds.copy()


class ListEnum(list, enum.Enum):
    def __new__(cls, *args):
            assert len(args) == 2
            try:
                inds = list(args[0])
            except TypeError:
                inds = [args[0], ]
            units = args[1]

            obj = list.__new__(cls)
            obj._value_ = inds
            obj.extend(inds)
            obj.units = units

            return obj

    def __init__(self, *args):
        pass

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        elif len(self.value) == 1:
            return self.value[0] == other
        elif len(self.value) == len(other):
            return self.value == other
        return NotImplemented()

    @classmethod
    def get_num_states(cls):
        n_states = -1
        for s in dir(cls):
            if s[0] == '_':
                continue
            v = getattr(cls, s)
            if max(v) > n_states:
                n_states = max(v)
        return n_states + 1


class v_smap(ListEnum):
    lat = (0, 'rad')
    lon = (1, 'rad')
    alt_wgs84 = (2, 'm')
    alt_msl = (3, 'm')
    alt_agl = (47, 'm')
    ned_pos = ([4, 5, 6], 'm')
    ned_vel = ([7, 8, 9], 'm/s')
    ned_accel = ([10, 11, 12], 'm/s^2')
    pitch = (13, 'rad')
    roll = (14, 'rad')
    yaw = (15, 'rad')
    body_vel = ([16, 17, 18], 'm/s')
    body_accel = ([19, 20, 21], 'm/s^2')
    body_rot_rate = ([22, 23, 24], 'rad/s')
    body_rot_accel = ([25, 26, 27], 'rad/s^2')
    dyn_pres = (28, 'Pa')
    airspeed = (29, 'm/s')
    mach = (30, '')
    aoa = (31, 'rad')
    aoa_rate = (32, 'rad/s')
    sideslip_ang = (33, 'rad')
    sideslip_rate = (34, 'rad/s')
    gnd_trk = (35, 'rad')
    fp_ang = (36, 'rad')
    gnd_speed = (37, 'm/s')
    dcm_earth2body = ([38, 39, 40, 41, 42, 43, 44, 45, 46], '')

    @classmethod
    def _get_ordered_key(cls, key, append_ind):
        lst = []
        for attr_str in dir(cls):
            if attr_str[0] == '_':
                continue

            attr = getattr(cls, attr_str)
            multi = len(attr.value) > 1
            is_dcm = multi and 'dcm' in attr.name
            for ii, jj in enumerate(attr.value):
                name = getattr(attr, key)
                if append_ind:
                    if is_dcm:
                        r, c = np.unravel_index([ii], (3, 3))
                        name += '_{:d}{:d}'.format(r.item(), c.item())
                    elif multi:
                        name += '_{:d}'.format(ii)

                lst.append((jj, name))
        lst.sort(key=lambda x: x[0])
        return tuple([x[1] for x in lst])

    @classmethod
    def get_ordered_names(cls):
        return cls._get_ordered_key('name', True)

    @classmethod
    def get_ordered_units(cls):
        return cls._get_ordered_key('units', False)


class e_smap(ListEnum):
    temp = (0, 'K')
    speed_of_sound = (1, 'm/s')
    pressure = (2, 'Pa')
    density = (3, 'kg/m^3')
    gravity = ([4, 5, 6], 'm/s^2')
    mag_field = ([7, 8, 9], 'uT')
    terrain_alt_wgs84 = (10, 'm')


class Vehicle:
    __slots__ = ('state', 'params', 'ref_lat', 'ref_lon')

    def __init__(self, params):
        self.state = np.nan * np.ones(v_smap.get_num_states())
        self.params = params

        self.ref_lat = np.nan
        self.ref_lon = np.nan

    def _get_dcm_earth2body(self):
        return self.state[v_smap.dcm_earth2body].reshape((3, 3))

    def set_dcm_earth2body(self, mat):
        self.state[v_smap.dcm_earth2body] = mat.flatten()

    def _calc_aero_force_mom(self, dyn_pres, body_vel):
        mom = np.zeros(3)
        xy_spd = np.linalg.norm(body_vel[0:2])
        if xy_spd < 1e-6:
            inc_ang = 0
        else:
            inc_ang = np.arctan(body_vel[2] / xy_spd)

        lut_npts = len(self.params.geo.front_area_m2)
        front_area = np.interp(inc_ang, np.linspace(-np.pi / 2, np.pi / 2, lut_npts),
                               self.params.geo.front_area_m2)

        vel_mag = np.linalg.norm(body_vel)
        if np.abs(vel_mag) < 1e-6:
            force = np.zeros(3)
        else:
            force = -body_vel / vel_mag * front_area * dyn_pres * self.params.aero.cd

        return force.ravel(), mom

    def _calc_grav_force_mom(self, gravity, dcm_earth2body):
        mom = np.zeros(3)
        force = dcm_earth2body @ (gravity * self.params.mass.mass_kg).reshape((3, 1))

        return force.ravel(), mom

    def _calc_prop_force_mom(self, motor_cmds):
        # motor model
        m_thrust = -np.polynomial.Polynomial(self.params.prop.poly_thrust[-1::-1])(motor_cmds)
        m_torque = -np.polynomial.Polynomial(self.params.prop.poly_torque[-1::-1])(motor_cmds)
        m_torque = np.sum(m_torque * np.array(self.params.motor.dir))

        # thrust to moment
        motor_mom = np.zeros(3)
        for ii, m_pos in enumerate(np.array(self.params.motor.pos_m)):
            dx = m_pos[0] - self.params.mass.cg_m[0]
            dy = m_pos[1] - self.params.mass.cg_m[1]
            motor_mom[0] += dy * m_thrust[ii]
            motor_mom[1] += -dx * m_thrust[ii]

        motor_mom[2] += m_torque

        # calc force
        force = np.zeros(3)
        force[2] = np.min([0, np.sum(m_thrust).item()])

        return force, motor_mom

    def _calc_force_mom(self, gravity, motor_cmds):
        a_f, a_m = self._calc_aero_force_mom(self.state[v_smap.dyn_pres],
                                             self.state[v_smap.body_vel])
        g_f, g_m = self._calc_grav_force_mom(gravity, self._get_dcm_earth2body())
        p_f, p_m = self._calc_prop_force_mom(motor_cmds)
        return (a_f + g_f + p_f, a_m + g_m + p_m)

    def _calc_pqr_dot(self, pqr, inertia, inertia_dot, moments):
        term0 = (inertia_dot @ pqr).ravel()
        term1 = np.cross(pqr.ravel(), (inertia @ pqr).ravel())

        term2 = moments - term0 - term1
        pqr_dot = term2.reshape((1, 3)) @ np.linalg.inv(inertia).T

        return pqr_dot.ravel()

    def _calc_uvw_dot(self, uvw, pqr, ned_accel):
        return ned_accel - np.cross(pqr, uvw)

    def _calc_eul_dot(self, eul, pqr):
        s_roll = np.sin(eul[0])
        c_roll = np.cos(eul[0])
        t_pitch = np.tan(eul[1])
        sc_pitch = 1 / np.cos(eul[1])
        R = np.array([[1, s_roll * t_pitch, c_roll * t_pitch],
                      [0, c_roll, -s_roll],
                      [0, s_roll * sc_pitch, c_roll * sc_pitch]])
        return R @ pqr

    def eul_to_dcm(self, r1, r2, r3):
        c1 = np.cos(r1)
        s1 = np.sin(r1)
        R1 = np.array([[c1, s1, 0],
                       [-s1, c1, 0],
                       [0, 0, 1]])
        c2 = np.cos(r2)
        s2 = np.sin(r2)
        R2 = np.array([[c2, 0, -s2],
                       [0, 1, 0],
                       [s2, 0, c2]])
        c3 = np.cos(r3)
        s3 = np.sin(r3)
        R3 = np.array([[1, 0, 0],
                       [0, c3, s3],
                       [0, -s3, c3]])

        return R3 @ R2 @ R1

    def _six_dof_model(self, force, mom, dt):
        ned_accel = force / self.params.mass.mass_kg

        body_rot_accel = self._calc_pqr_dot(self.state[v_smap.body_rot_rate].reshape((3, 1)),
                                            np.array(self.params.mass.inertia_kgm2),
                                            np.zeros((3, 3)), mom)

        # integrator to get body rotation rate
        r = integrate.ode(lambda t, x: self._calc_pqr_dot(x.reshape((3, 1)),
                                                          np.array(self.params.mass.inertia_kgm2),
                                                          np.zeros((3, 3)), mom))
        r.set_integrator('dopri5')
        r.set_initial_value(self.state[v_smap.body_rot_rate])
        r.integrate(dt)
        if not r.successful():
            raise RuntimeError('Integration of body rotation rate failed.')
        body_rot_rate = r.y

        body_accel = self._calc_uvw_dot(self.state[v_smap.body_vel], body_rot_rate,
                                        ned_accel)
        # integrator to get body velocity
        r = integrate.ode(lambda t, x: self._calc_uvw_dot(x, body_rot_rate, ned_accel))
        r.set_integrator('dopri5')
        r.set_initial_value(self.state[v_smap.body_vel])
        r.integrate(dt)
        if not r.successful():
            raise RuntimeError('Integration of body acceleration failed.')
        body_vel = r.y

        # integration to get euler angles
        r = integrate.ode(lambda t, x: self._calc_eul_dot(x, body_rot_rate))
        r.set_integrator('dopri5')
        eul_inds = v_smap.roll + v_smap.pitch + v_smap.yaw
        r.set_initial_value(self.state[eul_inds])
        r.integrate(dt)
        if not r.successful():
            raise RuntimeError('Integration of body rotation rate failed.')
        roll = r.y[0]
        pitch = r.y[1]
        yaw = r.y[2]

        # get dcm
        dcm_earth2body = self.eul_to_dcm(yaw, pitch, roll)

        # get ned vel and pos
        ned_vel = (dcm_earth2body.T @ body_vel).ravel()
        ned_pos = self.state[v_smap.ned_pos] + dt * ned_vel

        return (ned_vel, ned_pos, roll, pitch, yaw, dcm_earth2body, body_vel,
                body_rot_rate, body_rot_accel, body_accel, ned_accel)

    def calc_derived_states(self, dt, terrain_alt_wgs84, density, speed_of_sound,
                            ned_vel, ned_pos, body_vel):
        gnd_trk = np.arctan2(ned_vel[1], ned_vel[0])
        gnd_speed = np.linalg.norm(ned_vel[0:2])
        fp_ang = np.arctan2(-ned_vel[2], gnd_speed)

        dyn_pres = 0.5 * density * np.sum(body_vel * body_vel)

        aoa = np.arctan2(body_vel[2], body_vel[0])
        airspeed = np.linalg.norm(body_vel)
        if np.abs(airspeed) <= 1e-6:
            sideslip_ang = 0
        else:
            sideslip_ang = np.arcsin(body_vel[1] / airspeed)

        aoa_rate = (aoa - self.state[v_smap.aoa]) / dt
        sideslip_rate = (sideslip_ang - self.state[v_smap.sideslip_ang]) / dt

        mach = airspeed / speed_of_sound

        lla = ned_to_LLA(ned_pos.reshape((3, 1)), self.ref_lat, self.ref_lon,
                         terrain_alt_wgs84)
        lat = lla[0]
        lon = lla[1]
        alt_wgs84 = lla[2]

        alt_agl = alt_wgs84 - terrain_alt_wgs84
        alt_msl = wgs84.convert_wgs_to_msl(lla[0], lla[1], lla[2])

        return (gnd_trk, gnd_speed, fp_ang, dyn_pres, aoa, airspeed, sideslip_ang,
                aoa_rate, sideslip_rate, mach, lat, lon, alt_wgs84, alt_agl, alt_msl)

    def step(self, dt, terrain_alt_wgs84, gravity, density, speed_of_sound,
             motor_cmds):
        # gravity = np.array([0, 0, 9.81])
        force, mom = self._calc_force_mom(gravity, motor_cmds)

        (ned_vel, ned_pos, roll, pitch, yaw, dcm_earth2body, body_vel,
         body_rot_rate, body_rot_accel, body_accel,
         ned_accel) = self._six_dof_model(force, mom, dt)

        (gnd_trk, gnd_speed, fp_ang, dyn_pres, aoa, airspeed, sideslip_ang,
         aoa_rate, sideslip_rate, mach, lat, lon, alt_wgs84,
         alt_agl, alt_msl) = self.calc_derived_states(dt, terrain_alt_wgs84,
                                                      density, speed_of_sound,
                                                      ned_vel, ned_pos, body_vel)

        # update state
        self.state[v_smap.ned_vel] = ned_vel
        self.state[v_smap.ned_pos] = ned_pos
        self.state[v_smap.roll] = roll
        self.state[v_smap.pitch] = pitch
        self.state[v_smap.yaw] = yaw
        self.set_dcm_earth2body(dcm_earth2body)
        self.state[v_smap.body_vel] = body_vel
        self.state[v_smap.body_rot_rate] = body_rot_rate
        self.state[v_smap.body_rot_accel] = body_rot_accel
        self.state[v_smap.body_accel] = body_accel
        self.state[v_smap.ned_accel] = ned_accel
        self.state[v_smap.gnd_trk] = gnd_trk
        self.state[v_smap.gnd_speed] = gnd_speed
        self.state[v_smap.fp_ang] = fp_ang
        self.state[v_smap.dyn_pres] = dyn_pres
        self.state[v_smap.aoa] = aoa
        self.state[v_smap.aoa_rate] = aoa_rate
        self.state[v_smap.airspeed] = airspeed
        self.state[v_smap.sideslip_ang] = sideslip_ang
        self.state[v_smap.sideslip_rate] = sideslip_rate
        self.state[v_smap.mach] = mach
        self.state[v_smap.lat] = lat
        self.state[v_smap.lon] = lon
        self.state[v_smap.alt_wgs84] = alt_wgs84
        self.state[v_smap.alt_agl] = alt_agl
        self.state[v_smap.alt_msl] = alt_msl


class Environment:
    def __init__(self):
        self.state = np.nan * np.ones(e_smap.get_num_states())

    def _lower_atmo(self, alt_km):
        """This code is extracted from the version hosted on https://www.pdas.com/atmos.html
        of Public Domain Aerospace Software."""
        REARTH = 6356.7523  # polar radius of the earth, kilometers
        GZERO = 9.80665  # sea level accel. of gravity, m/s^2
        MOLWT_ZERO = 28.9644  # molecular weight of air at sea level
        RSTAR = 8314.32  # perfect gas constant, N-m/(kmol-K)
        GMR = 1000 * GZERO * MOLWT_ZERO / RSTAR  # hydrostatic constant, kelvins/km

        TEMP0 = 288.15  # sea-level temperature, kelvins
        PRES0 = 101325.0  # sea-level pressure, Pa
        DENSITY0 = 1.225  # sea-level density, kg/cu.m
        ASOUND0 = 340.294  # speed of sound at S.L. m/sec

        htab = [0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852]
        ttab = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
        ptab = [1.0, 2.2336110E-1, 5.4032950E-2, 8.5666784E-3, 1.0945601E-3,
                6.6063531E-4, 3.9046834E-5, 3.68501E-6]
        gtab = [-6.5, 0.0, 1.0, 2.8, 0, -2.8, -2.0, 0.0]

        # convert from geometric (height above geoid) to geopotential altitude
        h = alt_km * REARTH / (alt_km + REARTH)

        # binary search for atmo layer
        i = 0
        j = len(htab)
        while j > i + 1:
            k = (i + j) // 2
            if h < htab[k]:
                j = k
            else:
                i = k
        tgrad = gtab[i]  # temp. gradient of local layer
        tbase = ttab[i]  # base  temp. of local layer
        deltah = h - htab[i]  # height above local base
        tlocal = tbase + tgrad * deltah  # local temperature
        theta = tlocal / ttab[0]  # temperature ratio

        if 0.0 == tgrad:
            delta = ptab[i] * np.exp(-GMR * deltah / tbase).item()
        else:
            delta = ptab[i] * (tbase / tlocal)**(GMR / tgrad)
        sigma = delta / theta

        return (sigma * DENSITY0, delta * PRES0, theta * TEMP0,
                ASOUND0 * np.sqrt(theta))

    def _atmo(self, alt_msl):
        alt_km = alt_msl / 1000.0
        if alt_km < 86:
            return self._lower_atmo(alt_km)
        else:
            raise NotImplementedError('Upper atmosphere model (>86 km) not implemented.')

    def step(self, lat, lon, alt_wgs84, alt_msl):
        density, pres, temp, spd_snd = self._atmo(alt_msl)
        gravity = wgs84.calc_gravity(lat, alt_wgs84).ravel()

        # update state
        self.state[e_smap.temp] = temp
        self.state[e_smap.speed_of_sound] = spd_snd
        self.state[e_smap.pressure] = pres
        self.state[e_smap.density] = density
        self.state[e_smap.gravity] = gravity


class SimpleMultirotor(DynamicsBase):
    state_names = v_smap.get_ordered_names()
    state_units = v_smap.get_ordered_units()
    state_map = v_smap

    def __init__(self, params_file, env=None, effector=None, egm_bin_file=None, **kwargs):
        super().__init__(**kwargs)

        self._eff_req_init = effector is None
        if self._eff_req_init:
            self.effector = Effector()
        else:
            self.effector = effector

        self._env_req_init = env is None
        if self._env_req_init:
            self.env = Environment()
        else:
            self.env = env

        with open(self.validate_params_file(params_file), 'r') as fin:
            v_params = yaml.load(fin)

        self.vehicle = Vehicle(v_params)

        if egm_bin_file is not None:
            wgs84.init_egm_lookup_table(egm_bin_file)

    def validate_params_file(self, params_file):
        if os.pathsep in params_file:
            cf = params_file
        else:
            cf = os.path.join(os.getcwd(), params_file)
            if not os.path.isfile(cf):
                cf = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                  params_file)
                if not os.path.isfile(cf):
                    raise RuntimeError('Failed to find config file {}'.format(params_file))

        return cf

    def propagate_state(self, desired_motor_cmds, dt):
        motor_cmds = self.effector.step(desired_motor_cmds)
        self.env.step(self.vehicle.state[v_smap.lat], self.vehicle.state[v_smap.lon],
                      self.vehicle.state[v_smap.alt_wgs84], self.vehicle.state[v_smap.alt_msl])
        self.vehicle.step(dt, self.env.state[e_smap.terrain_alt_wgs84],
                          self.env.state[e_smap.gravity], self.env.state[e_smap.density],
                          self.env.state[e_smap.speed_of_sound], motor_cmds)

        return self.vehicle.state.copy()

    def set_initial_conditions(self, ned_pos, body_vel, eul_deg, body_rot_rate,
                               ref_lat_deg, ref_lon_deg, terrain_alt_wgs84,
                               ned_mag_field):
        if self._env_req_init:
            self.env.state[e_smap.mag_field] = ned_mag_field.flatten()
            self.env.state[e_smap.terrain_alt_wgs84] = terrain_alt_wgs84

        self.vehicle.state[v_smap.ned_pos] = ned_pos.flatten()
        self.vehicle.state[v_smap.body_vel] = body_vel.flatten()
        eul_rad = eul_deg * d2r
        self.vehicle.state[v_smap.roll] = eul_rad[0]
        self.vehicle.state[v_smap.pitch] = eul_rad[1]
        self.vehicle.state[v_smap.yaw] = eul_rad[2]
        dcm_earth2body = self.vehicle.eul_to_dcm(eul_rad[2], eul_rad[1],
                                                 eul_rad[0])
        self.vehicle.state[v_smap.ned_vel] = (dcm_earth2body.T
                                              @ body_vel.reshape((3, 1))).flatten()
        self.vehicle.set_dcm_earth2body(dcm_earth2body)
        self.vehicle.state[v_smap.body_rot_rate] = body_rot_rate.flatten()
        self.vehicle.state[v_smap.body_rot_accel] = 0
        self.vehicle.state[v_smap.body_accel] = 0
        self.vehicle.state[v_smap.body_accel[2]] = 9.81
        self.vehicle.state[v_smap.ned_accel] = 0
        self.vehicle.ref_lat = ref_lat_deg * d2r
        self.vehicle.ref_lon = ref_lon_deg * d2r

        lla = ned_to_LLA(ned_pos.reshape((3, 1)), ref_lat_deg * d2r, ref_lon_deg * d2r,
                         terrain_alt_wgs84)
        self.vehicle.state[v_smap.lat] = lla[0]
        self.vehicle.state[v_smap.lon] = lla[1]
        self.vehicle.state[v_smap.alt_wgs84] = lla[2]
        self.vehicle.state[v_smap.alt_msl] = wgs84.convert_wgs_to_msl(lla[0],
                                                                      lla[1],
                                                                      lla[2])

        # initialize the remaining environment state by calling step
        if self._env_req_init:
            self.env.step(self.vehicle.state[v_smap.lat], self.vehicle.state[v_smap.lon],
                          self.vehicle.state[v_smap.alt_wgs84], self.vehicle.state[v_smap.alt_msl])

        # get remaining vehicle derived states
        (gnd_trk, gnd_speed, fp_ang, dyn_pres, aoa, airspeed, sideslip_ang,
         _, _, mach, _, _, _,
         alt_agl, _) = self.vehicle.calc_derived_states(1, terrain_alt_wgs84,
                                                        self.env.state[e_smap.density],
                                                        self.env.state[e_smap.speed_of_sound],
                                                        self.vehicle.state[v_smap.ned_vel],
                                                        ned_pos, body_vel)

        self.vehicle.state[v_smap.gnd_trk] = gnd_trk
        self.vehicle.state[v_smap.gnd_speed] = gnd_speed
        self.vehicle.state[v_smap.fp_ang] = fp_ang
        self.vehicle.state[v_smap.dyn_pres] = dyn_pres
        self.vehicle.state[v_smap.aoa] = aoa
        self.vehicle.state[v_smap.aoa_rate] = 0
        self.vehicle.state[v_smap.airspeed] = airspeed
        self.vehicle.state[v_smap.sideslip_ang] = sideslip_ang
        self.vehicle.state[v_smap.sideslip_rate] = 0
        self.vehicle.state[v_smap.mach] = mach
        self.vehicle.state[v_smap.alt_agl] = alt_agl


class SimpleLAGERSuper(SimpleMultirotor):

    _s2us = 1000000.0
    wp_xy_scale = 1e7

    def __init__(self, params_file=None, **kwargs):
        if params_file is None:
            params_file = 'lager_super.yaml'

        self.num_waypoints = 0

        self._control_model = None
        self._sysData = super_bind.SysData()
        self._sensorData = super_bind.SensorData()
        self._navData = super_bind.NavData()
        self._telemData = super_bind.TelemData()
        self._vmsData = super_bind.VmsData()
        self._reached_wp_on_prev = False
        self._last_gps_upd_time = -np.inf
        self._gps_update_rate_hz = 5

        self._sensorData.inceptor.ch[4] = 990  # pos hold mode

        super().__init__(params_file, **kwargs)

        if self.control_model is None:
            self.control_model = super_bind.Autocode()

    @property
    def control_model(self):
        return self._control_model

    @control_model.setter
    def control_model(self, val):
        self._control_model = val

        if val is None:
            return

        try:
            self._control_model.initialize()
        except AttributeError:
            warn('Failed to call initialize function for control model')
            self._control_model = None

    @property
    def dt(self):
        return 0.01

    @dt.setter
    def dt(self, val):
        raise RuntimeError('dt is readonly')

    @property
    def state(self):
        return self.vehicle.state

    @property
    def waypoints(self):
        if len(self._telemData.flight_plan) > 0:
            return self._telemData.flight_plan[:self.num_waypoints]
        else:
            return []

    @property
    def home_alt_wgs84_m(self):
        return self._navData.home_alt_wgs84_m

    @property
    def home_lat_rad(self):
        return self._navData.home_lat_rad

    @property
    def home_lon_rad(self):
        return self._navData.home_lon_rad

    @property
    def waypoint_reached(self):
        return self._vmsData.waypoint_reached

    @property
    def finished_waypoints(self):
        return self._telemData.current_waypoint >= self.num_waypoints

    @property
    def current_waypoint(self):
        return self._telemData.current_waypoint

    @property
    def waypoint_radius(self):
        return self._telemData.param[10]

    @waypoint_radius.setter
    def waypoint_radius(self, val):
        self._telemData.param[10] = float(val)

    def parse_waypoint_file(self, file):
        waypoints = [super_bind.MissionItem()] * super_bind.NUM_FLIGHT_PLAN_POINTS
        home = super_bind.MissionItem()
        num_waypoints = 0
        with open(file, 'r') as fin:
            for line_no, line in enumerate(fin):
                if line_no == 0:
                    continue
                cols = [s.strip() for s in line.split()]
                missionItem = super_bind.MissionItem()
                missionItem.frame = int(cols[2])
                missionItem.cmd = int(cols[3])
                missionItem.param1 = float(cols[4])
                missionItem.param2 = float(cols[5])
                missionItem.param3 = float(cols[6])
                missionItem.param4 = float(cols[7])
                if missionItem.cmd == 16:
                    mult = self.wp_xy_scale
                missionItem.x = int(float(cols[8]) * mult)
                missionItem.y = int(float(cols[9]) * mult)
                missionItem.z = float(cols[10])
                missionItem.autocontinue = cols[11] == '1'

                if line_no == 1:
                    home = missionItem
                else:
                    waypoints[num_waypoints] = missionItem
                    num_waypoints += 1

        self.num_waypoints = num_waypoints
        return waypoints, num_waypoints, home

    def upload_waypoints(self, waypoints, num_waypoints=None, current_waypoint=0):
        if num_waypoints is None:
            self.num_waypoints = len(waypoints)
        else:
            self.num_waypoints = num_waypoints

        waypoint_lst = waypoints
        if len(waypoint_lst) < super_bind.NUM_FLIGHT_PLAN_POINTS:
            rem = super_bind.NUM_FLIGHT_PLAN_POINTS - len(waypoint_lst)
            waypoint_lst.extend([super_bind.MissionItem() for ii in range(rem)])
        elif len(waypoint_lst) > super_bind.NUM_FLIGHT_PLAN_POINTS:
            waypoint_lst = waypoint_lst[:super_bind.NUM_FLIGHT_PLAN_POINTS]

        self._telemData.waypoints_updated = True
        self._telemData.num_waypoints = self.num_waypoints
        if self.num_waypoints > 0:
            self._sensorData.inceptor.ch[4] = 1811  # waypoint follow mode
            self._telemData.flight_plan = waypoint_lst  # NOTE: can only update full list at once
        else:
            self._sensorData.inceptor.ch[4] = 990  # pos hold
        self._telemData.current_waypoint = int(current_waypoint)

    def set_initial_conditions(self, ned_pos, body_vel, eul_deg, body_rot_rate,
                               ned_mag_field, home_wp=None, waypoint_file=None,
                               ref_lat_deg=None, ref_lon_deg=None, terrain_alt_wgs84=None):
        if home_wp is not None or waypoint_file is not None:
            if waypoint_file is not None:
                waypoints, num_waypoints, home_wp = self.parse_waypoint_file(waypoint_file)
                self.upload_waypoints(waypoints, num_waypoints=num_waypoints)

            ref_lat_deg = home_wp.x / self.wp_xy_scale
            ref_lon_deg = home_wp.y / self.wp_xy_scale
            terrain_alt_wgs84 = wgs84.convert_msl_to_wgs(ref_lat_deg, ref_lon_deg,
                                                         home_wp.z)

        elif ref_lat_deg is not None and ref_lon_deg is not None and terrain_alt_wgs84 is not None:
            pass
        else:
            raise RuntimeError('Invalid input combination: Must specify reference position.')

        self._navData.home_lat_rad = ref_lat_deg * d2r
        self._navData.home_lon_rad = ref_lon_deg * d2r
        self._navData.home_alt_wgs84_m = terrain_alt_wgs84

        super().set_initial_conditions(ned_pos, body_vel, eul_deg, body_rot_rate,
                                       ref_lat_deg, ref_lon_deg, terrain_alt_wgs84,
                                       ned_mag_field)

    def update_sys_data(self, tt):
        self._sysData.frame_time_us = int(tt * self._s2us)
        self._sysData.sys_time_us = int(tt * self._s2us)

    def emulate_accel(self):
        accel = self.vehicle.state[self.state_map.body_accel].copy().reshape((3, 1))
        accel += self.vehicle._get_dcm_earth2body() @ self.env.state[e_smap.gravity].reshape((3, 1))
        # TODO: add noise
        return accel.ravel()

    def emulate_gnss(self):
        self._sensorData.gnss.healthy = True
        self._sensorData.gnss.fix = super_bind.GnssFix.GNSS_FIX_3D
        self._sensorData.gnss.num_sats = 16
        self._sensorData.gnss.horz_acc_m = 1.5
        self._sensorData.gnss.vert_acc_m = 5.5
        self._sensorData.gnss.vel_acc_mps = 0.05
        self._sensorData.gnss.hdop = 0.7
        self._sensorData.gnss.vdop = 0.7
        self._sensorData.gnss.track_acc_rad = 2 * d2r

        # TODO: add noise to lat/lon
        self._sensorData.gnss.lat_rad = self.vehicle.state[self.state_map.lat].copy()
        self._sensorData.gnss.lon_rad = self.vehicle.state[self.state_map.lon].copy()
        self._sensorData.gnss.alt_wgs84_m = self.vehicle.state[self.state_map.alt_wgs84].copy()
        self._sensorData.gnss.alt_msl_m = self.vehicle.state[self.state_map.alt_msl].copy()

        # TODO: add noise to vel
        self._sensorData.gnss.ned_vel_mps = self.vehicle.state[self.state_map.ned_vel].copy()
        self._sensorData.gnss.track_rad = np.arctan2(self._sensorData.gnss.ned_vel_mps[1],
                                                     self._sensorData.gnss.ned_vel_mps[0])
        self._sensorData.gnss.spd_mps = np.linalg.norm(self._sensorData.gnss.ned_vel_mps[0:2])

    def update_sensor_data(self, tt):
        # update inceptor data
        self._sensorData.inceptor.new_data = True
        self._sensorData.inceptor.lost_frame = False
        self._sensorData.inceptor.failsafe = False
        self._sensorData.inceptor.ch17 = False
        self._sensorData.inceptor.ch18 = False

        self._sensorData.inceptor.ch[0] = 991
        self._sensorData.inceptor.ch[1] = 991
        self._sensorData.inceptor.ch[2] = 991
        self._sensorData.inceptor.ch[3] = 991
        self._sensorData.inceptor.ch[6] = 1811

        # update imu
        self._sensorData.imu.accel_mps2 = self.emulate_accel()
        self._sensorData.imu.gyro_radps = self.vehicle.state[self.state_map.body_rot_rate].copy()

        # update GNSS data
        self._sensorData.gnss.new_data = (tt - self._last_gps_upd_time) >= (1 / self._gps_update_rate_hz)
        if self._sensorData.gnss.new_data:
            # emulate gps reciever
            self.emulate_gnss()
            self._last_gps_upd_time = tt

        # update Pressure data
        self._sensorData.static_pres.new_data = True
        self._sensorData.static_pres.healthy = True
        self._sensorData.static_pres.pres_pa = self.env.state[e_smap.pressure].copy()
        self._sensorData.static_pres.die_temp_c = 23

        # update power module
        self._sensorData.power_module.voltage_v = 12 * 4.2  # nCells * v/cell

    def update_nav_data(self):
        self._navData.nav_initialized = True

        # TODO: do the init conds need to be subtracted here?
        self._navData.pitch_rad = self.vehicle.state[self.state_map.pitch].copy()
        self._navData.roll_rad = self.vehicle.state[self.state_map.roll].copy()

        self._navData.heading_rad = self.vehicle.state[self.state_map.yaw].copy()
        self._navData.alt_wgs84_m = self.vehicle.state[self.state_map.alt_wgs84].copy()
        self._navData.alt_msl_m = self.vehicle.state[self.state_map.alt_msl].copy()
        self._navData.alt_rel_m = self.vehicle.state[self.state_map.alt_agl].copy()
        self._navData.flight_path_rad = self.vehicle.state[self.state_map.fp_ang].copy()
        self._navData.ned_pos_m = self.vehicle.state[self.state_map.ned_pos].copy()
        self._navData.ned_vel_mps = self.vehicle.state[self.state_map.ned_vel].copy()
        self._navData.lat_rad = self.vehicle.state[self.state_map.lat].copy()
        self._navData.lon_rad = self.vehicle.state[self.state_map.lon].copy()

    @property
    def desired_motor_cmds(self):
        return self._vmsData.pwm.cmd[0:self.vehicle.params.motor.num_motors].copy()

    @desired_motor_cmds.setter
    def desired_motor_cmds(self, val):
        raise RuntimeError('desired_motor_cmds is readonly')

    def update_fmu_states(self, tt):
        self.update_sys_data(tt)
        self.update_sensor_data(tt)
        self.update_nav_data()

        # rising edge trigger
        if self._vmsData.waypoint_reached and not self._reached_wp_on_prev:
            self._telemData.current_waypoint += 1
        self._reached_wp_on_prev = self._vmsData.waypoint_reached

    def propagate_state(self, tt):
        self.update_fmu_states(tt)

        self._control_model.step(self._sysData, self._sensorData, self._navData,
                                 self._telemData, self._vmsData)

        self._telemData.waypoints_updated = False

        return super().propagate_state(self._vmsData.pwm.cmd[0:self.vehicle.params.motor.num_motors], self.dt)
