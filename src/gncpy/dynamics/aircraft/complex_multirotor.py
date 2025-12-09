"""Implements a complex multi-rotor dynamics model with arbitrary motor orientations.

This module provides a comprehensive multirotor dynamics model supporting:
- Arbitrary motor thrust directions (not limited to vertical)
- Bidirectional thrust control (-1 to +1 command range)
- Quaternion-based attitude representation (avoids gimbal lock)
- Proper reaction torque modeling with sigma (rotation direction)

Key Features:
- Motors can be tilted, vectored, or point in any direction
- Thrust can reverse along thrust_dir vector (useful for tilting rotors)
- Uses passive rotation quaternion convention (body-to-NED)
- Body-frame accelerations and angular accelerations available in state vector

Frame Conventions:
- Quaternion q: passive rotation (body-to-NED), use with q*v*q_conj
- For NED-to-body: use quat_conjugate(q) or quat_to_dcm(q)
- State vector contains both NED and body-frame quantities
"""

import numpy as np
from scipy import integrate

from gncpy.dynamics.aircraft.simple_multirotor import (
    SimpleMultirotor,
    MotorParams as SimpleMotorParams,
    AircraftParams as SimpleAircraftParams,
    Vehicle as SimpleVehicle,
    v_smap,
    ListEnum,
    yaml,
)
import gncpy.math as gmath


class ComplexMotorParams(SimpleMotorParams):
    """Motor parameters for complex multirotor with arbitrary motor orientations.

    Attributes
    ----------
    pos_m : list
        Each element is a list of the position of the motor in meters (body frame).
    dir : list
        Each element is +/-1 indicating the direction the motor spins (sigma).
        Positive (+1) means CCW rotation about the thrust axis when viewed from the
        direction the thrust vector points. Negative (-1) means CW rotation.
        This follows right-hand rule: thumb along thrust axis, fingers show rotation.
    thrust_dir : list
        Each element is a 3-element list representing the unit vector direction
        of the motor thrust in the body frame. Will be normalized if not already unit length.
    """

    def __init__(self):
        super().__init__()
        self.thrust_dir = []

    def validate_and_normalize(self):
        """Validate parameters and normalize thrust direction vectors.

        Ensures that thrust_dir has the same length as pos_m and dir, and
        normalizes all thrust direction vectors to unit length.

        Raises
        ------
        ValueError
            If the parameter lists have inconsistent lengths.
        """
        n_motors = len(self.pos_m)
        if len(self.dir) != n_motors:
            raise ValueError(
                f"Length mismatch: pos_m has {n_motors} motors but dir has {len(self.dir)}"
            )
        if len(self.thrust_dir) != n_motors:
            raise ValueError(
                f"Length mismatch: pos_m has {n_motors} motors but thrust_dir has {len(self.thrust_dir)}"
            )

        # Normalize thrust direction vectors
        for i in range(n_motors):
            thrust_vec = np.array(self.thrust_dir[i])
            if thrust_vec.shape != (3,):
                raise ValueError(
                    f"Motor {i} thrust_dir must be a 3-element vector, got shape {thrust_vec.shape}"
                )
            mag = np.linalg.norm(thrust_vec)
            if mag < np.finfo(float).eps:
                raise ValueError(f"Motor {i} thrust_dir has zero magnitude")
            self.thrust_dir[i] = (thrust_vec / mag).tolist()


class ComplexAircraftParams(SimpleAircraftParams):
    """Aircraft parameters for complex multirotor.

    Attributes
    ----------
    aero : :class:`.AeroParams`
        Aerodynamic parameters.
    mass : :class:`.MassParams`
        Mass parameters.
    geo : :class:`.GeoParams`
        Geometric parameters
    prop : :class:`.PropParams`
        Propeller parameters.
    motor : :class:`.ComplexMotorParams`
        Complex motor parameters with thrust directions.
    """

    def __init__(self):
        super().__init__()
        self.motor = ComplexMotorParams()


yaml.register_class(ComplexMotorParams)
yaml.register_class(ComplexAircraftParams)


class v_smap_quat(ListEnum):
    """Enum for the vehicle state using quaternions instead of Euler angles.

    This state map replaces the Euler angles (roll, pitch, yaw) and DCM with
    a quaternion representation to avoid gimbal lock. The quaternion is stored
    in scalar-first format: [qw, qx, qy, qz] where qw is the scalar component.
    """

    lat = (0, "rad")
    lon = (1, "rad")
    alt_wgs84 = (2, "m")
    alt_msl = (3, "m")
    alt_agl = (47, "m")
    ned_pos = ([4, 5, 6], "m")
    ned_vel = ([7, 8, 9], "m/s")
    ned_accel = ([10, 11, 12], "m/s^2")
    quat = ([13, 14, 15, 16], "")  # [qw, qx, qy, qz] - scalar first
    body_vel = ([17, 18, 19], "m/s")
    body_accel = ([20, 21, 22], "m/s^2")
    body_rot_rate = ([23, 24, 25], "rad/s")
    body_rot_accel = ([26, 27, 28], "rad/s^2")
    dyn_pres = (29, "Pa")
    airspeed = (30, "m/s")
    mach = (31, "")
    aoa = (32, "rad")
    aoa_rate = (33, "rad/s")
    sideslip_ang = (34, "rad")
    sideslip_rate = (35, "rad/s")
    gnd_trk = (36, "rad")
    fp_ang = (37, "rad")
    gnd_speed = (38, "m/s")
    # Note: DCM is removed, computed from quaternion when needed
    # Euler angles can be computed from quaternion when needed

    @classmethod
    def _get_ordered_key(cls, key, append_ind):
        lst = []
        for attr_str in dir(cls):
            if attr_str[0] == "_":
                continue

            attr = getattr(cls, attr_str)
            multi = len(attr.value) > 1
            is_quat = multi and "quat" in attr.name
            for ii, jj in enumerate(attr.value):
                name = getattr(attr, key)
                if append_ind:
                    if is_quat:
                        quat_names = ["qw", "qx", "qy", "qz"]
                        name += "_" + quat_names[ii]
                    elif multi:
                        name += "_{:d}".format(ii)

                lst.append((jj, name))
        lst.sort(key=lambda x: x[0])
        return tuple([x[1] for x in lst])

    @classmethod
    def get_ordered_names(cls):
        """Get the state names in the order they appear in the vector including indices."""
        return cls._get_ordered_key("name", True)

    @classmethod
    def get_ordered_units(cls):
        """Get a list of units for each state in the vector in sorted order."""
        return cls._get_ordered_key("units", False)


class ComplexVehicle(SimpleVehicle):
    """Implements a vehicle with arbitrary motor thrust orientations and quaternion attitude.

    This extends the simple vehicle to support:
    - Motors that can point in any direction, not just purely vertical
    - Quaternion-based attitude representation to avoid gimbal lock

    The moment contribution from each motor includes both the moment arm
    cross product with thrust and the reaction torque from rotor spin.

    Attributes
    ----------
    state : numpy array
        State of the aircraft using v_smap_quat.
    params : :class:`.ComplexAircraftParams`
        Parameters of the aircraft including motor thrust directions.
    ref_lat : float
        Reference latitude in radians (for converting to NED)
    ref_lon : float
        Reference longitude in radians (for converting to NED)
    takenoff : bool
        Flag indicating if the vehicle has taken off yet.
    """

    def __init__(self, params):
        """Initialize an object.

        Parameters
        ----------
        params : :class:`.ComplexAircraftParams`
            Parameters of the aircraft.
        """
        # Don't call parent init - we need different state size
        self.params = params
        self.state = np.nan * np.ones(v_smap_quat.get_num_states())
        self.ref_lat = np.nan
        self.ref_lon = np.nan
        self.takenoff = False

        # Validate and normalize motor parameters
        params.motor.validate_and_normalize()

    def _get_dcm_earth2body(self):
        """Get DCM from current quaternion state."""
        q = self.state[v_smap_quat.quat]
        return gmath.quat_to_dcm(q)

    def set_quaternion(self, q):
        """Set the quaternion in the state vector.

        Parameters
        ----------
        q : numpy array
            Quaternion [qw, qx, qy, qz] to assign
        """
        self.state[v_smap_quat.quat] = gmath.quat_normalize(q)

    def _calc_force_mom(self, gravity, motor_cmds):
        """Calculate forces and moments using quaternion-based gravity transformation.

        This overrides the parent method to use quaternion rotation directly
        with arbitrary motor thrust directions.

        Parameters
        ----------
        gravity : numpy array
            Gravity vector in NED frame (m/s^2)
        motor_cmds : numpy array
            Motor commands in normalized range [-1, 1] for bidirectional thrust

        Returns
        -------
        tuple
            (total_force, total_moment) in body frame
        """
        # Get aerodynamic forces
        a_f, a_m = self._calc_aero_force_mom(
            self.state[v_smap_quat.dyn_pres], self.state[v_smap_quat.body_vel]
        )

        # Transform gravity from NED to body frame using quaternion
        # Note: The quaternion represents body-to-NED rotation (passive rotation),
        # so we use its conjugate to rotate from NED to body
        q = self.state[v_smap_quat.quat]
        q_inv = gmath.quat_conjugate(q)  # NED-to-body rotation
        gravity_ned = gravity * self.params.mass.mass_kg
        gravity_body = gmath.quat_rotate_vector(q_inv, gravity_ned)
        g_f = gravity_body
        g_m = np.zeros(3)  # Gravity produces no moment about CG

        # Get propulsion forces with arbitrary thrust directions
        p_f, p_m = self._calc_prop_force_mom(motor_cmds)

        if not self.takenoff:
            # Check if upward thrust exceeds gravity
            self.takenoff = np.linalg.norm(p_f) > np.linalg.norm(g_f)

        if self.takenoff:
            return (a_f + g_f + p_f, a_m + g_m + p_m)
        else:
            return np.zeros(a_f.shape), np.zeros(a_m.shape)

    def _calc_prop_force_mom(self, motor_cmds):
        """Calculate propulsion forces and moments with arbitrary motor orientations.

        This overrides the simple multirotor implementation to handle motors
        that can point in arbitrary directions and support bidirectional thrust.

        Bidirectional Thrust Support:
        - Motor commands range from -1 (full reverse) to +1 (full forward)
        - Thrust magnitude: sign(cmd) x polynomial(|cmd|)
        - This allows thrust reversal along the thrust_dir vector
        - Useful for tilting rotors, vectored thrust, or reversible propellers

        For each motor:
        - Thrust magnitude preserves sign: can be positive or negative
        - Thrust force vector = thrust_magnitude x thrust_dir (direction reverses with sign)
        - Moment from thrust = (motor_pos - cg) x thrust_force
        - Reaction torque = -sigma x |torque_magnitude| x thrust_dir
          * Sigma +1 (CCW about thrust axis): body experiences -thrust_dir torque
          * Sigma -1 (CW about thrust axis): body experiences +thrust_dir torque

        Parameters
        ----------
        motor_cmds : numpy array
            Commands to motors, range [-1, 1] for bidirectional thrust.
            Simple multirotor uses [0, 1] for upward-only thrust.

        Returns
        -------
        force : numpy array
            Total force in body frame (3,).
        motor_mom : numpy array
            Total moment in body frame (3,).
        """
        # Motor model - compute thrust with sign preservation for bidirectional thrust
        # Thrust: sign(cmd) x polynomial(|cmd|) allows reversal along thrust_dir
        m_thrust_mag = np.sign(motor_cmds) * np.polynomial.Polynomial(
            self.params.prop.poly_thrust[-1::-1]
        )(np.abs(motor_cmds))

        # Torque: uses absolute value since direction is determined by sigma
        m_torque_mag = np.polynomial.Polynomial(self.params.prop.poly_torque[-1::-1])(
            np.abs(motor_cmds)
        )

        # Initialize force and moment
        total_force = np.zeros(3)
        total_moment = np.zeros(3)

        cg = np.array(self.params.mass.cg_m)

        # Iterate through each motor
        for i in range(self.params.motor.num_motors):
            # Thrust direction unit vector (already normalized in validate_and_normalize)
            thrust_dir = np.array(self.params.motor.thrust_dir[i])

            # Motor position
            motor_pos = np.array(self.params.motor.pos_m[i])

            # Sigma: direction of rotation (+1 = CCW, -1 = CW) about thrust axis
            sigma = self.params.motor.dir[i]

            # Thrust force vector
            thrust_force = m_thrust_mag[i] * thrust_dir

            # Accumulate total force
            total_force += thrust_force

            # Moment from thrust: r x F where r is position relative to CG
            r = motor_pos - cg
            moment_from_thrust = np.cross(r, thrust_force)

            # Reaction torque: -sigma * magnitude * thrust_direction
            # (negative because it opposes rotor spin)
            reaction_torque = -sigma * m_torque_mag[i] * thrust_dir

            # Accumulate total moment
            total_moment += moment_from_thrust + reaction_torque

        return total_force, total_moment

    def _six_dof_model(self, force, mom, dt):
        """Six degree of freedom model using quaternion dynamics.

        This overrides the parent's Euler-angle-based dynamics to use
        quaternions, avoiding gimbal lock.

        Parameters
        ----------
        force : numpy array
            Force vector in body frame (N)
        mom : numpy array
            Moment vector in body frame (N-m)
        dt : float
            Time step (s)

        Returns
        -------
        tuple
            (ned_vel, ned_pos, quaternion, body_vel, body_rot_rate,
             body_rot_accel, body_accel, ned_accel)
        """

        def ode_quat(t, x, f, m):
            """ODE for quaternion-based 6-DOF dynamics.

            State vector x:
            [0:3]   NED position (m)
            [3:6]   Body velocity (m/s)
            [6:10]  Quaternion [qw, qx, qy, qz] (scalar first)
            [10:13] Body angular rates (rad/s)
            """
            # Extract state components
            # ned_pos = x[0:3]
            body_vel = x[3:6]
            quat = x[6:10]
            omega = x[10:13]

            # Normalize quaternion
            quat = gmath.quat_normalize(quat)

            # State derivatives
            xdot = np.zeros(13)

            # NED position derivative (rotate body velocity to NED frame)
            # Passive rotation quaternion (body-to-NED) convention
            xdot[0:3] = gmath.quat_rotate_vector(quat, body_vel)

            # Body velocity derivative (specific force + Coriolis term in rotating frame)
            # Correct sign: -omega x v_B for rotating frame transport theorem
            xdot[3:6] = f / self.params.mass.mass_kg - np.cross(omega, body_vel)

            # Quaternion derivative (kinematics)
            # qdot = 0.5 * Omega(omega) * q for scalar-first [qw, qx, qy, qz]
            qw, qx, qy, qz = quat
            wx, wy, wz = omega

            xdot[6] = 0.5 * (-wx * qx - wy * qy - wz * qz)  # qw_dot
            xdot[7] = 0.5 * (wx * qw + wz * qy - wy * qz)  # qx_dot
            xdot[8] = 0.5 * (wy * qw - wz * qx + wx * qz)  # qy_dot
            xdot[9] = 0.5 * (wz * qw + wy * qx - wx * qy)  # qz_dot

            # Angular velocity derivative (Euler's equation)
            J = np.array(self.params.mass.inertia_kgm2)
            xdot[10:13] = np.linalg.inv(J) @ (m - np.cross(omega, J @ omega))

            return xdot

        # Setup integrator
        r = integrate.ode(ode_quat).set_integrator("dopri5").set_f_params(force, mom)

        # Initial state vector
        x0 = np.concatenate(
            (
                self.state[v_smap_quat.ned_pos].flatten(),
                self.state[v_smap_quat.body_vel].flatten(),
                self.state[v_smap_quat.quat].flatten(),
                self.state[v_smap_quat.body_rot_rate].flatten(),
            )
        )

        r.set_initial_value(x0, 0)
        y = r.integrate(dt)

        if not r.successful():
            raise RuntimeError("Integration failed.")

        # Extract integrated state
        ned_pos = y[0:3]
        body_vel = y[3:6]
        quat = gmath.quat_normalize(y[6:10])
        body_rot_rate = y[10:13]

        # NED velocity (rotate body velocity to NED frame)
        # Passive rotation quaternion (body-to-NED) convention
        ned_vel = gmath.quat_rotate_vector(quat, body_vel)

        # Compute accelerations from derivatives
        xdot = ode_quat(dt, y, force, mom)
        body_accel = xdot[3:6]
        body_rot_accel = xdot[10:13]
        ned_accel = gmath.quat_rotate_vector(quat, body_accel)

        return (
            ned_vel,
            ned_pos,
            quat,
            body_vel,
            body_rot_rate,
            body_rot_accel,
            body_accel,
            ned_accel,
        )

    def step(self, dt, terrain_alt_wgs84, gravity, density, speed_of_sound, motor_cmds):
        """Perform one update step for the vehicle.

        This overrides the parent to use quaternion dynamics and the new state map.

        Parameters
        ----------
        dt : float
            Delta time since last update.
        terrain_alt_wgs84 : float
            Altitude of the terrain relative to WGS-84 model in meters.
        gravity : numpy array
            gravity vector.
        density : float
            Density of the atmosphere.
        speed_of_sound : float
            Speed of sound in m/s.
        motor_cmds : numpy array
            Commands to the motors in normalized range.
        """
        force, mom = self._calc_force_mom(gravity, motor_cmds)

        (
            ned_vel,
            ned_pos,
            quat,
            body_vel,
            body_rot_rate,
            body_rot_accel,
            body_accel,
            ned_accel,
        ) = self._six_dof_model(force, mom, dt)

        (
            gnd_trk,
            gnd_speed,
            fp_ang,
            dyn_pres,
            aoa,
            airspeed,
            sideslip_ang,
            aoa_rate,
            sideslip_rate,
            mach,
            lat,
            lon,
            alt_wgs84,
            alt_agl,
            alt_msl,
        ) = self.calc_derived_states(
            dt, terrain_alt_wgs84, density, speed_of_sound, ned_vel, ned_pos, body_vel
        )

        # Update state with quaternion
        self.state[v_smap_quat.ned_vel] = ned_vel
        self.state[v_smap_quat.ned_pos] = ned_pos
        self.state[v_smap_quat.quat] = quat
        self.state[v_smap_quat.body_vel] = body_vel
        self.state[v_smap_quat.body_rot_rate] = body_rot_rate
        self.state[v_smap_quat.body_rot_accel] = body_rot_accel
        self.state[v_smap_quat.body_accel] = body_accel
        self.state[v_smap_quat.ned_accel] = ned_accel
        self.state[v_smap_quat.gnd_trk] = gnd_trk
        self.state[v_smap_quat.gnd_speed] = gnd_speed
        self.state[v_smap_quat.fp_ang] = fp_ang
        self.state[v_smap_quat.dyn_pres] = dyn_pres
        self.state[v_smap_quat.aoa] = aoa
        self.state[v_smap_quat.aoa_rate] = aoa_rate
        self.state[v_smap_quat.airspeed] = airspeed
        self.state[v_smap_quat.sideslip_ang] = sideslip_ang
        self.state[v_smap_quat.sideslip_rate] = sideslip_rate
        self.state[v_smap_quat.mach] = mach
        self.state[v_smap_quat.lat] = lat
        self.state[v_smap_quat.lon] = lon
        self.state[v_smap_quat.alt_wgs84] = alt_wgs84
        self.state[v_smap_quat.alt_agl] = alt_agl
        self.state[v_smap_quat.alt_msl] = alt_msl


class ComplexMultirotor(SimpleMultirotor):
    """Implements a complex multi-rotor with arbitrary motor orientations and quaternion attitude.

    This class extends SimpleMultirotor to support:
    - Motors that can point in any direction (not just vertical)
    - Motor rotation direction (sigma) for proper reaction torque modeling
    - Quaternion-based attitude representation to avoid gimbal lock

    Attributes
    ----------
    effector : :class:`.Effector`
        Effectors for the vehicle.
    env : :class:`.Environment`
        Environment the vehicle is in.
    vehicle : :class:`.ComplexVehicle`
        Complex vehicle class with arbitrary motor thrust directions and quaternion dynamics.
    """

    state_names = v_smap_quat.get_ordered_names()
    """List of vehicle state names."""

    state_units = v_smap_quat.get_ordered_units()
    """List of vehicle state units."""

    state_map = v_smap_quat
    """Map of states to indices with units."""

    def __init__(
        self,
        params_file,
        env=None,
        effector=None,
        egm_bin_file=None,
        library_dir=None,
        **kwargs,
    ):
        """Initialize an object.

        Parameters
        ----------
        params_file : string
            Full path to the config file. The config file should use
            ComplexAircraftParams format with thrust_dir specified for each motor.
        env : :class:`.Environment`, optional
            Environment for the vehicle. The default is None.
        effector : :class:`.Effector`, optional
            Effector for the vehicle. The default is None.
        egm_bin_file : string, optional
            Full path to the binary file for the EGM model. The default is None.
        library_dir : string, optional
            Default directory to look for config files. The default is None.
        **kwargs : dict
            Additional arguments for the parent class.
        """
        # Don't call parent __init__ yet - we need to override vehicle creation
        # Instead, manually initialize what we need
        super(SimpleMultirotor, self).__init__(**kwargs)

        if library_dir is None:
            import pathlib
            import os

            self.library_config_dir = os.path.join(
                pathlib.Path(__file__).parent.resolve(),
            )
        else:
            self.library_config_dir = library_dir

        self._eff_req_init = effector is None
        if self._eff_req_init:
            from gncpy.dynamics.aircraft.simple_multirotor import Effector

            self.effector = Effector()
        else:
            self.effector = effector

        self._env_req_init = env is None
        if self._env_req_init:
            from gncpy.dynamics.aircraft.simple_multirotor import Environment

            self.env = Environment()
        else:
            self.env = env

        # Load parameters as ComplexAircraftParams
        with open(self.validate_params_file(params_file), "r") as fin:
            v_params = yaml.load(fin)

        # Create ComplexVehicle instead of simple Vehicle
        self.vehicle = ComplexVehicle(v_params)

        if egm_bin_file is not None:
            import gncpy.wgs84 as wgs84

            wgs84.init_egm_lookup_table(egm_bin_file)

    def set_initial_conditions(
        self,
        ned_pos,
        body_vel,
        eul_deg,
        body_rot_rate,
        ref_lat_deg,
        ref_lon_deg,
        terrain_alt_wgs84,
        ned_mag_field,
        body_accel=None,
        body_rot_accel=None,
    ):
        """Set initial conditions for the state using quaternion representation.

        Parameters
        ----------
        ned_pos : numpy array
            Body position in NED frame.
        body_vel : numpy array
            Velocity of the body in body frame.
        eul_deg : numpy array
            Initial attitude in degrees and yaw, pitch, roll order.
        body_rot_rate : numpy array
            Initial body rotation rate (rad/s).
        ref_lat_deg : float
            Reference latitude in degrees.
        ref_lon_deg : float
            Reference longitude in degrees.
        terrain_alt_wgs84 : float
            Altitude of the terrain relative to WGS-84 model in meters.
        ned_mag_field : numpy array
            Local magnetic field vector in NED frame and uT.
        body_accel : numpy array, optional
            Initial body acceleration in m/s^2. Default is zeros.
        body_rot_accel : numpy array, optional
            Initial body rotational acceleration in rad/s^2. Default is zeros.
        """
        from gncpy.coordinate_transforms import ned_to_LLA
        from gncpy.dynamics.aircraft.simple_multirotor import e_smap
        import gncpy.wgs84 as wgs84

        d2r = np.pi / 180.0

        # Convert Euler angles to quaternion
        yaw_rad = eul_deg[0] * d2r
        pitch_rad = eul_deg[1] * d2r
        roll_rad = eul_deg[2] * d2r
        quat = gmath.euler_to_quat(roll_rad, pitch_rad, yaw_rad)

        # Set vehicle state
        self.vehicle.state[v_smap_quat.ned_pos] = ned_pos.flatten()
        self.vehicle.state[v_smap_quat.body_vel] = body_vel.flatten()
        self.vehicle.state[v_smap_quat.quat] = quat
        self.vehicle.state[v_smap_quat.body_rot_rate] = body_rot_rate.flatten()

        # Set reference location
        self.vehicle.ref_lat = ref_lat_deg * d2r
        self.vehicle.ref_lon = ref_lon_deg * d2r

        # Set accelerations (default to zeros if not provided)
        if body_accel is None:
            body_accel = np.zeros(3)
        if body_rot_accel is None:
            body_rot_accel = np.zeros(3)
        self.vehicle.state[v_smap_quat.body_accel] = body_accel.flatten()
        self.vehicle.state[v_smap_quat.body_rot_accel] = body_rot_accel.flatten()

        # Compute NED velocity and acceleration from body frame using quaternion
        # The quaternion represents body-to-NED rotation (passive rotation)
        self.vehicle.state[v_smap_quat.ned_vel] = gmath.quat_rotate_vector(
            quat, body_vel
        )
        self.vehicle.state[v_smap_quat.ned_accel] = gmath.quat_rotate_vector(
            quat, body_accel
        )

        # Compute LLA from NED
        lla = ned_to_LLA(
            ned_pos.reshape((3, 1)),
            self.vehicle.ref_lat,
            self.vehicle.ref_lon,
            terrain_alt_wgs84,
        )
        self.vehicle.state[v_smap_quat.lat] = lla[0]
        self.vehicle.state[v_smap_quat.lon] = lla[1]
        self.vehicle.state[v_smap_quat.alt_wgs84] = lla[2]
        self.vehicle.state[v_smap_quat.alt_msl] = wgs84.convert_wgs_to_msl(
            lla[0], lla[1], lla[2]
        )

        # Initialize environment state
        if self._env_req_init:
            self.env.state[e_smap.mag_field] = ned_mag_field.flatten()
            self.env.state[e_smap.terrain_alt_wgs84] = terrain_alt_wgs84

        # Step environment to compute atmosphere/gravity
        self.env.step(
            self.vehicle.state[v_smap_quat.lat],
            self.vehicle.state[v_smap_quat.lon],
            self.vehicle.state[v_smap_quat.alt_wgs84],
            self.vehicle.state[v_smap_quat.alt_msl],
        )

        # Compute all derived states
        (
            gnd_trk,
            gnd_speed,
            fp_ang,
            dyn_pres,
            aoa,
            airspeed,
            sideslip_ang,
            _,
            _,
            mach,
            _,
            _,
            _,
            alt_agl,
            _,
        ) = self.vehicle.calc_derived_states(
            1,
            terrain_alt_wgs84,
            self.env.state[e_smap.density],
            self.env.state[e_smap.speed_of_sound],
            self.vehicle.state[v_smap_quat.ned_vel],
            ned_pos,
            body_vel,
        )

        self.vehicle.state[v_smap_quat.gnd_trk] = gnd_trk
        self.vehicle.state[v_smap_quat.gnd_speed] = gnd_speed
        self.vehicle.state[v_smap_quat.fp_ang] = fp_ang
        self.vehicle.state[v_smap_quat.dyn_pres] = dyn_pres
        self.vehicle.state[v_smap_quat.aoa] = aoa
        self.vehicle.state[v_smap_quat.aoa_rate] = 0
        self.vehicle.state[v_smap_quat.airspeed] = airspeed
        self.vehicle.state[v_smap_quat.sideslip_ang] = sideslip_ang
        self.vehicle.state[v_smap_quat.sideslip_rate] = 0
        self.vehicle.state[v_smap_quat.mach] = mach
        self.vehicle.state[v_smap_quat.alt_agl] = alt_agl

        self._env_req_init = False

    def propagate_state(self, timestep, state, u=None, state_args=None, ctrl_args=None):
        """Propagates the state forward 1 timestep.

        This overrides the base class to use quaternion-based vehicle dynamics.

        Parameters
        ----------
        timestep : float
            Timestep to use for propagation. If None, uses self.dt.
        state : numpy array
            Full vehicle state vector.
        u : numpy array, optional
            Motor commands in range [-1, 1] for bidirectional thrust.
            The default is None.
        state_args : tuple, optional
            Not used. The default is None.
        ctrl_args : tuple, optional
            Not used. The default is None.

        Returns
        -------
        numpy array
            Next state as column vector.
        """
        from gncpy.dynamics.aircraft.simple_multirotor import e_smap

        # Use provided timestep or default
        dt = self.dt if timestep is None else timestep

        # Set the vehicle state from input
        self.vehicle.state = state.ravel().copy()

        # Get motor commands
        if u is None:
            raise ValueError("Motor commands (u) must be provided")
        motor_cmds = self.effector.step(u.ravel(), dt)

        # Update environment
        self.env.step(
            self.vehicle.state[v_smap_quat.lat],
            self.vehicle.state[v_smap_quat.lon],
            self.vehicle.state[v_smap_quat.alt_wgs84],
            self.vehicle.state[v_smap_quat.alt_msl],
        )

        # Step vehicle dynamics
        self.vehicle.step(
            dt,
            self.env.state[e_smap.terrain_alt_wgs84],
            self.env.state[e_smap.gravity],
            self.env.state[e_smap.density],
            self.env.state[e_smap.speed_of_sound],
            motor_cmds,
        )

        return self.vehicle.state.copy().reshape((-1, 1))
