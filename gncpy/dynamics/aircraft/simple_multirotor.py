import numpy as np
from scipy import integrate
import enum
from warnings import warn
from ruamel.yaml import YAML

from gncpy.dynamics.basic import DynamicsBase
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


class PropParams:
    def __init__(self):
        self.poly_thrust = []
        self.poly_torque = []


class MotorParams:
    def __init__(self):
        self.pos_m = []


class AircraftParams:
    def __init__(self):
        self.aero = AeroParams()
        self.mass = MassParams()
        self.prop = PropParams()
        self.motor = MotorParams()


yaml.register_class(AeroParams)
yaml.register_class(MassParams)
yaml.register_class(PropParams)
yaml.register_class(MotorParams)
yaml.register_class(AircraftParams)


class Effector:
    def step(self, input_cmds):
        return input_cmds.copy()


class ListEnum(list, enum.Enum):
    def __new__(cls, *args):
            assert len(args) == 2
            print(args)
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
        for attr in dir(cls):
            if attr[0] == '_':
                continue

            multi = len(attr.value) > 1
            is_dcm = multi and 'dcm' in attr.name
            for ii in attr.value:
                name = getattr(attr, key)
                if append_ind:
                    if is_dcm:
                        r, c = np.unravel_index([ii], (3, 3))
                        name += '_{:d}{:d}'.format(r.item(), c.item())
                    elif multi:
                        name += '_{:d}'.format(ii)

                lst.append((ii, name))
        lst.sort(key=lambda x: x[0])
        return lst

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
        n_states = -1
        for s in dir(v_smap):
            if s[0:2] == '__' or s[0] == '_':
                continue
            v = getattr(v_smap, s)
            if max(v) > n_states:
                n_states = max(v)
        self.state = np.nan * np.ones(n_states).reshape((-1, 1))
        self.params = params

        self.ref_lat = np.nan
        self.ref_lon = np.nan

    def _get_dcm_earth2body(self):
        return self.state[v_smap.dcm_earth2body].reshape((3, 3))

    def set_dcm_earth2body(self, mat):
        self.state[v_smap.dcm_earth2body] = mat.flatten()

    def _calc_aero_force_mom(self, dyn_pres, body_vel):
        mom = np.zeros(3)
        inc_ang = np.arctan(body_vel[2] / np.linalg.norm(body_vel[0:2]))

        lut_npts = len(self.params.geo.front_area_m2)
        front_area = np.interp(inc_ang, np.linspace(-np.pi / 2, np.pi / 2, lut_npts),
                               self.params.geo.front_area_m2)

        force = -body_vel / np.linalg.norm(body_vel) * front_area * dyn_pres * self.params.aero.cd

        return force.ravel(), mom

    def _calc_grav_force_mom(self, gravity, dcm_earth2body):
        mom = np.zeros(3)
        force = dcm_earth2body @ (gravity * self.params.mass.mass_kg).reshape((3, 1))

        return force.ravel(), mom

    def _calc_prop_force_mom(self, motor_cmds):
        # motor model
        m_thrust = -np.polynomial.Polynomial(self.params.motor.poly_thrust[-1::-1])(motor_cmds)
        m_torque = -np.polynomial.Polynomial(self.params.motor.poly_torque[-1::-1])(motor_cmds)
        m_torque = np.sum(m_torque * np.array(self.params.motor.dir))

        # thrust to moment
        motor_mom = np.zeros(3)
        for ii, m_pos in enumerate(np.array(self.params.motor.pos_m).reshape((-1, 3))):
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
        term1 = np.cross(pqr, (inertia @ pqr).ravel())

        term2 = moments - term0 - term1
        pqr_dot = term2.reshape((1, 3)) @ np.linalg.inv(inertia)

        return pqr_dot.ravel()

    def _calc_uvw_dot(self, uvw, pqr, ned_accel):
        return ned_accel + np.cross(uvw, pqr)

    def _calc_eul_dot(self, eul, pqr):
        s_eul = np.sin(eul)
        c_eul = np.cos(eul)
        return np.array([pqr[0] + (pqr[1] * s_eul[0] + pqr[2] * c_eul[0]) * (s_eul[1] / c_eul[1]),
                         pqr[1] * c_eul[0] - pqr[2] * s_eul[0],
                         (pqr[1] * s_eul[0] + pqr[2] * c_eul[0]) / c_eul[1]])

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

        body_rot_accel = self._calc_pqr_dot(self.state[v_smap.body_rot_rate],
                                            self.params.mass.inertia_kgm2,
                                            np.zeros((3, 3)), mom)

        # integrator to get body rotation rate
        r = integrate.ode(lambda t, x: self._calc_pqr_dot(x.reshape((3, 1)),
                                                          self.params.mass.inertia_kgm2,
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

        aoa = np.atan2(body_vel[2], body_vel[0])
        airspeed = np.linalg.norm(body_vel)
        sideslip_ang = np.arcsin(body_vel[1] / airspeed)

        aoa_rate = (aoa - self.state[v_smap.aoa]) / dt
        sideslip_rate = (sideslip_ang - self.state[v_smap.sideslip_ang]) / dt

        mach = airspeed / speed_of_sound

        lla = ned_to_LLA(ned_pos, self.ref_lat, self.ref_lon, -terrain_alt_wgs84)
        lat = lla[0]
        lon = lla[1]
        alt_wgs84 = lla[2]

        alt_agl = alt_wgs84 - terrain_alt_wgs84
        alt_msl = wgs84.convert_wgs_to_msl(lla[0], lla[1], lla[2])

        return (gnd_trk, gnd_speed, fp_ang, dyn_pres, aoa, airspeed, sideslip_ang,
                aoa_rate, sideslip_rate, mach, lat, lon, alt_wgs84, alt_agl, alt_msl)

    def step(self, dt, terrain_alt_wgs84, gravity, density, speed_of_sound,
             motor_cmds):
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
        n_states = -1
        for s in dir(v_smap):
            if s[0:2] == '__' or s[0] == '_':
                continue
            v = getattr(v_smap, s)
            if max(v) > n_states:
                n_states = max(v)

        self.state = np.nan * np.ones(n_states)

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
            delta = ptab[i] * np.pow(tbase / tlocal, GMR / tgrad).item()
        sigma = delta / theta

        return (sigma * DENSITY0, delta * PRES0, theta * TEMP0,
                ASOUND0 * np.sqrt(theta))

    def _atmo(self, alt_msl):
        alt_km = 1000 * alt_msl
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


class GenericMultirotor(DynamicsBase):
    state_names = v_smap.get_ordered_names()
    state_units = v_smap.get_ordered_units()

    def __init__(self, params_file, env=None, effector=None, egm_bin_file=None):
        super().__init__()

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

        with open(params_file, 'r') as fin:
            data = yaml.load(fin)

        self.vehicle = Vehicle(data)

        if egm_bin_file is not None:
            wgs84.init_egm_lookup_table(egm_bin_file)

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
        self.vehicle.state[v_smap.ned_accel] = 0
        self.vehicle.ref_lat = ref_lat_deg * d2r
        self.vehicle.ref_lon = ref_lon_deg * d2r

        lla = ned_to_LLA(ned_pos, ref_lat_deg * d2r, ref_lon_deg * d2r,
                         -terrain_alt_wgs84)
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


class LAGERSuper(GenericMultirotor):
    def __init__(self, params_file=None, **kwargs):
        if params_file is None:
            params_file = './lager_super.yaml'
        super().__init__(params_file, **kwargs)
