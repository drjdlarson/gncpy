"""Test omnicopter INDI controller with various maneuvers.

This script tests INDI control with:
    Test 1: Circular velocity tracking (zero angular rates)
    Test 2: 360 degree roll maneuver (zero velocity)
    Test 3: Velocity + attitude maneuver

Realistic Sensor Model:
    The simulation uses a realistic sensor configuration that mimics
    actual multirotor hardware:

    - Velocity: Simulates output from a state estimator (e.g., EKF fusing
      GPS and IMU), with realistic noise characteristics

    - Accelerometer: Direct body-frame acceleration measurement with
      typical IMU noise (SIGMA_ACCEL = 0.1 m/s²)

    - Gyroscope: Direct body-frame angular rate measurement with
      typical IMU noise (SIGMA_OMEGA = 0.01 rad/s)

    - Angular Acceleration: Computed via numerical differentiation of
      filtered gyroscope measurements (backward difference). This is
      realistic as most systems don't directly measure angular acceleration.
      An aggressive low-pass filter (5 Hz) is applied to mitigate noise
      amplification from the differentiation.

    All sensor measurements are low-pass filtered before use in the INDI
    controller to reduce noise amplification in the control loop.

Wind Disturbance:
    The simulation includes significant wind disturbances:
    - Constant wind: [5.0, 3.0, 0.5] m/s in NED frame
    - Sinusoidal gusts: ±[2.0, 1.5, 0.5] m/s amplitude
    - Aerodynamic drag forces computed from relative wind velocity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from gncpy.dynamics.aircraft.complex_multirotor import (
    ComplexMultirotor,
    v_smap_quat,
)
from gncpy.dynamics.aircraft.simple_multirotor import (
    Effector,
    e_smap,
    yaml,
)
import gncpy.math as gmath

from gncpy.control.INDI import INDI


# ============================================================================
# First-order motor dynamics effector with efficiency variation
# ============================================================================
class MotorDynamicsEffector(Effector):
    """First-order motor dynamics model with per-motor efficiency variation.

    Models motor response as a first-order lag:
        omega_dot_i = (1/tau_mot) * (omega_cmd,i - omega_i)

    Additionally applies per-motor efficiency scaling to simulate
    manufacturing variation and wear. This creates B-matrix mismatch
    between the LoFi model (used for control design) and HiFi truth.

    Motor efficiencies (hardcoded ±2% variation):
        Motor 0: 0.99 (-1%)
        Motor 1: 1.01 (+1%)
        Motor 2: 0.98 (-2%)
        Motor 3: 1.02 (+2%)
        Motor 4: 1.01 (+1%)
        Motor 5: 0.99 (-1%)
        Motor 6: 1.00 (0%)
        Motor 7: 0.98 (-2%)
    """

    # Hardcoded motor efficiency variation (±2% range)
    MOTOR_EFFICIENCY = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def __init__(self, num_motors, tau_mot, initial_state=None):
        self.num_motors = num_motors
        self.tau_mot = tau_mot

        if initial_state is not None:
            self.state = np.array(initial_state).flatten().copy()
        else:
            self.state = np.zeros(num_motors)

    def set_initial_state(self, initial_state):
        """Set the initial motor state."""
        self.state = np.array(initial_state).flatten().copy()

    def step(self, input_cmds, dt):
        """Propagate motor dynamics one timestep using exact solution.

        Applies motor efficiency scaling to simulate thrust variation.
        """
        input_cmds = np.array(input_cmds).flatten()

        # Apply per-motor efficiency scaling (simulates manufacturing variation)
        scaled_cmds = input_cmds * self.MOTOR_EFFICIENCY[: self.num_motors]

        alpha = np.exp(-dt / self.tau_mot)
        self.state = scaled_cmds + (self.state - scaled_cmds) * alpha
        return self.state.copy()


# ============================================================================
# Constants
# ============================================================================
DT = 0.001  # Timestep
TAU_MOT = 0.032  # Motor time constant

# Initial conditions
INITIAL_POSITION = np.array([0.0, 0.0, -10.0])  # NED (m)
INITIAL_VELOCITY = np.array([0.0, 0.0, 0.0])
INITIAL_ATTITUDE = np.array([0.0, 0.0, 0.0])  # Roll, pitch, yaw (deg)
INITIAL_ANGULAR_VELOCITY = np.array([0.0, 0.0, 0.0])

REF_LAT, REF_LON, TERRAIN_ALT = 34.0, -86.0, 0.0

# INDI gains
K_VEL = 5.0
K_OMEGA = 10.0

# Measurement noise - realistic sensor specifications
# These values simulate what actual sensors provide
SIGMA_VEL = 0.05  # m/s (velocity from state estimator/GPS)
SIGMA_ACCEL = 0.1  # m/s^2 (accelerometer noise - typical IMU spec)
SIGMA_OMEGA = 0.01  # rad/s (gyroscope noise - typical IMU spec)

# Optional measurement bias (set to 0 for no bias)
BIAS_VEL = np.array([0.0, 0.0, 0.0])  # m/s
BIAS_ACCEL = np.array([0.0, 0.0, 0.0])  # m/s^2
BIAS_OMEGA = np.array([0.0, 0.0, 0.0])  # rad/s

# Low-pass filter for measurements (reduces noise amplification in INDI)
FC_VEL = 5.0  # Hz (velocity filter cutoff)
FC_ACCEL = 20.0  # Hz (accelerometer filter cutoff)
FC_OMEGA = 20.0  # Hz (angular rate filter cutoff)
FC_ALPHA = 5.0  # Hz (angular acceleration derivative filter - aggressive to handle differentiation noise)

# Wind parameters (NED frame) - strong wind to test rejection
WIND_VELOCITY = np.array([8.0, 5.0, 1.0])  # m/s (constant wind in NED)
WIND_GUST_AMP = np.array([4.0, 3.0, 1.0])  # m/s (gust amplitude)
WIND_GUST_FREQ = np.array([0.3, 0.4, 0.5])  # Hz (gust frequencies)

np.random.seed(42)  # For reproducible noise


def compute_wind_force(wind_ned, vel_ned, quat, cd, frontal_area, air_density=1.225):
    """Compute aerodynamic drag force from wind.

    Parameters
    ----------
    wind_ned : ndarray
        Wind velocity in NED frame (m/s)
    vel_ned : ndarray
        Vehicle velocity in NED frame (m/s)
    quat : ndarray
        Vehicle quaternion [w, x, y, z]
    cd : float
        Drag coefficient
    frontal_area : ndarray
        Frontal areas in body frame [x, y, z, unused] (m^2)
    air_density : float
        Air density (kg/m^3)

    Returns
    -------
    force_body : ndarray
        Drag force in body frame (N)
    """
    # Relative wind velocity in NED (wind - vehicle_velocity)
    v_rel_ned = wind_ned - vel_ned

    # Convert to body frame
    from gncpy.math import quat_inverse, quat_to_dcm

    dcm_ned_to_body = quat_to_dcm(quat).T
    v_rel_body = dcm_ned_to_body @ v_rel_ned

    # Scale up frontal area for visibility (real frontal area is too small)
    # Using 10x the actual area to make wind effects visible
    effective_area = frontal_area[:3]

    # Drag force in body frame: F = 0.5 * rho * cd * A * v^2 * sign(v)
    force_body = np.zeros(3)
    for i in range(3):
        force_body[i] = (
            0.5
            * air_density
            * cd
            * effective_area[i]
            * v_rel_body[i]
            * np.abs(v_rel_body[i])
        )

    return force_body


# ============================================================================
# Setup dynamics and controller
# ============================================================================
print("=" * 70)
print("Setting up omnicopter dynamics and INDI controller")
print("=" * 70)

lofi_config_file = Path(__file__).parent / "omnicopter_config.yaml"
hifi_config_file = Path(__file__).parent / "omnicopter_config_hifi_copy.yaml"

# Load LoFi to get B0 matrix
lofi_dyn = ComplexMultirotor(str(lofi_config_file))
ned_mag = np.array([20.0, 5.0, 45.0])
lofi_dyn.set_initial_conditions(
    INITIAL_POSITION,
    INITIAL_VELOCITY,
    INITIAL_ATTITUDE,
    INITIAL_ANGULAR_VELOCITY,
    REF_LAT,
    REF_LON,
    TERRAIN_ALT,
    ned_mag,
)

gravity = lofi_dyn.env.state[e_smap.gravity]
print(f"Gravity: {gravity[2]:.4f} m/s^2")

# Build B0 matrix from LoFi model
mass = lofi_dyn.vehicle.params.mass.mass_kg
inertia = np.array(lofi_dyn.vehicle.params.mass.inertia_kgm2)
num_motors = lofi_dyn.vehicle.params.motor.num_motors
T_max = lofi_dyn.vehicle.params.prop.poly_thrust[0]

N = np.zeros((3, num_motors))
P = np.zeros((3, num_motors))
for i in range(num_motors):
    N[:, i] = lofi_dyn.vehicle.params.motor.thrust_dir[i]
    P[:, i] = lofi_dyn.vehicle.params.motor.pos_m[i]

B0_force = (1.0 / mass) * T_max * N

P_cross_N = np.zeros((3, num_motors))
for i in range(num_motors):
    P_cross_N[:, i] = np.cross(P[:, i], N[:, i])

J_inv = np.linalg.inv(inertia)
B0_moment = J_inv @ (T_max * P_cross_N)
B0 = np.vstack([B0_force, B0_moment])
B0_inv = np.linalg.pinv(B0)

print(f"B0 matrix: {B0.shape}")
print(f"Number of motors: {num_motors}")

# Compute hover commands for HiFi
g_mag = gravity[2]
desired_accel = np.array([0.0, 0.0, -g_mag, 0.0, 0.0, 0.0])
hover_cmds_lofi = B0_inv @ desired_accel

with open(hifi_config_file, "r") as f:
    hifi_params = yaml.load(f)
hifi_thrust_poly = hifi_params.prop.poly_thrust
c2 = hifi_thrust_poly[0]
lofi_thrust = T_max * hover_cmds_lofi
hover_cmds_hifi = np.sign(lofi_thrust) * np.sqrt(np.abs(lofi_thrust) / c2)

print(
    f"Hover commands (HiFi): [{hover_cmds_hifi.min():.3f}, {hover_cmds_hifi.max():.3f}]"
)

# Create HiFi dynamics with motor effector
motor_effector = MotorDynamicsEffector(
    num_motors=num_motors,
    tau_mot=TAU_MOT,
    initial_state=hover_cmds_hifi,
)

hifi_dyn = ComplexMultirotor(str(hifi_config_file), effector=motor_effector)

# Create INDI controller
K = np.diag([K_VEL, K_VEL, K_VEL, K_OMEGA, K_OMEGA, K_OMEGA])
indi_ctrl = INDI(omit_A=True)
indi_ctrl.set_state_model(dt=DT, K=K, B0=B0)

print(f"INDI gains: K_vel={K_VEL}, K_omega={K_OMEGA}")
print("Setup complete!\n")


# ============================================================================
# Test 1: Circular velocity tracking (zero angular rates)
# ============================================================================
print("=" * 70)
print("TEST 1: Circular Velocity Tracking")
print("=" * 70)

# Reset dynamics to hover
motor_effector.set_initial_state(hover_cmds_hifi)
hifi_dyn.set_initial_conditions(
    INITIAL_POSITION,
    INITIAL_VELOCITY,
    INITIAL_ATTITUDE,
    INITIAL_ANGULAR_VELOCITY,
    REF_LAT,
    REF_LON,
    TERRAIN_ALT,
    ned_mag,
)
hifi_dyn.vehicle.takenoff = True

# Simulation parameters for Test 1
SIM_TIME_1 = 10.0  # seconds
num_steps_1 = int(SIM_TIME_1 / DT)

# Circular velocity reference
V_CIRCLE = 1.0  # m/s
OMEGA_CIRCLE = 2.0 * np.pi / 8.0  # rad/s (8 second period)

print(f"Circle velocity: {V_CIRCLE} m/s")
print(f"Circle period: {2*np.pi/OMEGA_CIRCLE:.2f} s")
print(f"Simulation time: {SIM_TIME_1} s")

# Storage
time_hist_1 = np.zeros(num_steps_1)
body_vel_hist_1 = np.zeros((num_steps_1, 3))
body_vel_ref_hist_1 = np.zeros((num_steps_1, 3))
body_omega_hist_1 = np.zeros((num_steps_1, 3))
body_omega_ref_hist_1 = np.zeros((num_steps_1, 3))
ned_pos_hist_1 = np.zeros((num_steps_1, 3))
euler_hist_1 = np.zeros((num_steps_1, 3))
cmd_hist_1 = np.zeros((num_steps_1, num_motors))

# Initial conditions
cur_state = hifi_dyn.vehicle.state.copy()
cur_input = hover_cmds_hifi.copy()

# Initialize low-pass filter states for sensor measurements
vel_filt = np.zeros(3)
accel_filt = np.zeros(3)
omega_filt = np.zeros(3)
alpha_filt = np.zeros(3)  # Filtered angular acceleration

# Filter coefficients (first-order discrete low-pass)
alpha_vel = DT / (DT + 1.0 / (2.0 * np.pi * FC_VEL))
alpha_accel = DT / (DT + 1.0 / (2.0 * np.pi * FC_ACCEL))
alpha_omega = DT / (DT + 1.0 / (2.0 * np.pi * FC_OMEGA))
alpha_alpha = DT / (DT + 1.0 / (2.0 * np.pi * FC_ALPHA))

# Previous omega measurement for numerical differentiation
omega_prev = np.zeros(3)

# Load vehicle aerodynamic parameters for wind force calculation
cd = hifi_dyn.vehicle.params.aero.cd
frontal_area = np.array(hifi_dyn.vehicle.params.geo.front_area_m2)

print("Running Test 1 simulation with wind...")
print(
    f"Constant wind: [{WIND_VELOCITY[0]:.1f}, {WIND_VELOCITY[1]:.1f}, {WIND_VELOCITY[2]:.1f}] m/s NED"
)
print(
    f"Gust amplitude: ±[{WIND_GUST_AMP[0]:.1f}, {WIND_GUST_AMP[1]:.1f}, {WIND_GUST_AMP[2]:.1f}] m/s\n"
)

for ii in range(num_steps_1):
    tt = ii * DT
    time_hist_1[ii] = tt

    # Current true state
    body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
    body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
    body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
    quat = cur_state[v_smap_quat.quat].flatten()
    ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

    # Compute wind with gusts
    wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
    wind_total = WIND_VELOCITY + wind_gust

    # Compute wind drag force in body frame
    wind_force_body = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)

    # ====================================================================
    # REALISTIC SENSOR MODEL:
    # - Velocity: from state estimator (e.g., GPS + IMU fusion)
    # - Accelerometer: direct measurement with noise
    # - Gyroscope: direct measurement with noise
    # - Angular acceleration: numerically differentiated from gyro
    # ====================================================================

    # 1. Velocity measurement (simulates output from state estimator)
    vel_noise = np.random.normal(0, SIGMA_VEL, 3)
    body_vel_meas = body_vel_true + vel_noise + BIAS_VEL
    vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt

    # 2. Accelerometer measurement (direct sensor reading)
    accel_noise = np.random.normal(0, SIGMA_ACCEL, 3)
    body_accel_meas = body_accel_true + accel_noise + BIAS_ACCEL
    accel_filt = alpha_accel * body_accel_meas + (1.0 - alpha_accel) * accel_filt

    # 3. Gyroscope measurement (direct sensor reading)
    omega_noise = np.random.normal(0, SIGMA_OMEGA, 3)
    body_omega_meas = body_omega_true + omega_noise + BIAS_OMEGA
    omega_filt = alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt

    # 4. Angular acceleration via numerical differentiation
    # Using filtered omega to reduce noise amplification
    # Backward difference: alpha[k] = (omega[k] - omega[k-1]) / dt
    if ii == 0:
        # First timestep: use zero (or could use true value)
        alpha_meas = np.zeros(3)
    else:
        alpha_meas = (omega_filt - omega_prev) / DT

    # Apply aggressive low-pass filter to angular acceleration
    # (differentiation amplifies high-frequency noise)
    alpha_filt = alpha_alpha * alpha_meas + (1.0 - alpha_alpha) * alpha_filt

    # Store current omega for next differentiation step
    omega_prev = omega_filt.copy()

    # State and derivatives for INDI controller (what sensors provide)
    x = np.concatenate([vel_filt, omega_filt])
    x_dot = np.concatenate([accel_filt, alpha_filt])

    # Reference: circular velocity in body frame, zero angular rates
    vb_ref = np.array(
        [
            V_CIRCLE * np.cos(OMEGA_CIRCLE * tt),
            V_CIRCLE * np.sin(OMEGA_CIRCLE * tt),
            0.0,
        ]
    )
    omega_ref = np.zeros(3)
    ref = np.concatenate([vb_ref, omega_ref])

    # Reference derivative
    vb_ref_dot = np.array(
        [
            -V_CIRCLE * OMEGA_CIRCLE * np.sin(OMEGA_CIRCLE * tt),
            V_CIRCLE * OMEGA_CIRCLE * np.cos(OMEGA_CIRCLE * tt),
            0.0,
        ]
    )
    omega_ref_dot = np.zeros(3)
    ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

    # INDI control (unaware of wind)
    u_cmd = indi_ctrl.calculate_control(
        cur_time=tt,
        cur_state=x,
        cur_state_dot=x_dot,
        cur_input=cur_input,
        ref=ref,
        ref_dot=ref_dot,
    )
    u_cmd = np.clip(u_cmd, -1.0, 1.0)

    # Store data (store true values for analysis)
    body_vel_hist_1[ii, :] = body_vel_true
    body_vel_ref_hist_1[ii, :] = vb_ref
    body_omega_hist_1[ii, :] = body_omega_true
    body_omega_ref_hist_1[ii, :] = omega_ref
    ned_pos_hist_1[ii, :] = cur_state[v_smap_quat.ned_pos].flatten()
    cmd_hist_1[ii, :] = u_cmd

    # Convert quaternion to Euler angles for plotting
    roll, pitch, yaw = gmath.quat_to_euler(quat)
    euler_hist_1[ii, :] = np.rad2deg([roll, pitch, yaw])

    # Propagate dynamics with wind disturbance
    next_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()

    # Add wind force as external acceleration (F/m in body frame)
    mass = hifi_dyn.vehicle.params.mass.mass_kg
    wind_accel_body = wind_force_body / mass
    next_state[v_smap_quat.body_vel] += wind_accel_body * DT

    cur_state = next_state
    cur_input = motor_effector.state.copy()

print("Test 1 complete!")
print(
    f"Final velocity error: {np.linalg.norm(body_vel_hist_1[-1, :] - body_vel_ref_hist_1[-1, :]):.4f} m/s"
)
print(f"Final angular rate: {np.linalg.norm(body_omega_hist_1[-1, :]):.6f} rad/s\n")


# ============================================================================
# Plot Test 1 results
# ============================================================================
print("Plotting Test 1 results...")

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig1.suptitle(
    "Test 1: Circular Velocity Tracking (Zero Angular Rates)",
    fontsize=14,
    fontweight="bold",
)

# Body velocity tracking
ax = axes1[0, 0]
ax.plot(time_hist_1, body_vel_hist_1[:, 0], "r-", label="vb_x")
ax.plot(time_hist_1, body_vel_ref_hist_1[:, 0], "r--", alpha=0.7, label="vb_x ref")
ax.plot(time_hist_1, body_vel_hist_1[:, 1], "g-", label="vb_y")
ax.plot(time_hist_1, body_vel_ref_hist_1[:, 1], "g--", alpha=0.7, label="vb_y ref")
ax.plot(time_hist_1, body_vel_hist_1[:, 2], "b-", label="vb_z")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Velocity (m/s)")
ax.set_ylim([-1.3, 1.3])
ax.legend(loc="upper right", ncol=3, fontsize=8)
ax.grid(True, alpha=0.3)

# Body angular rate
ax = axes1[0, 1]
ax.plot(time_hist_1, np.rad2deg(body_omega_hist_1[:, 0]), "r-", label="p")
ax.plot(time_hist_1, np.rad2deg(body_omega_hist_1[:, 1]), "g-", label="q")
ax.plot(time_hist_1, np.rad2deg(body_omega_hist_1[:, 2]), "b-", label="r")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Angular Rate (deg/s)")
ax.set_ylim([-65, 65])
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Euler angles
ax = axes1[1, 0]
ax.plot(time_hist_1, euler_hist_1[:, 0], "r-", label="Roll")
ax.plot(time_hist_1, euler_hist_1[:, 1], "g-", label="Pitch")
ax.plot(time_hist_1, euler_hist_1[:, 2], "b-", label="Yaw")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Euler Angles (deg)")
ax.set_ylim([-180, 180])
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Motor commands
ax = axes1[1, 1]
for ii in range(num_motors):
    ax.plot(time_hist_1, cmd_hist_1[:, ii], alpha=0.7, label=f"u{ii}")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Motor Commands")
ax.set_ylim([-1.0, 1.0])
ax.legend(loc="upper right", ncol=4, fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent / "ValidationResults"
output_dir.mkdir(exist_ok=True)
output_file_1 = output_dir / "indi_test1_circular_velocity_wind_copy.png"
plt.savefig(output_file_1, dpi=150)
print(f"Test 1 figure saved to: {output_file_1}")


# ============================================================================
# Test 2: 360 degree roll maneuver (zero velocity)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: 360 Degree Roll Maneuver")
print("=" * 70)

# Reset dynamics to hover
motor_effector.set_initial_state(hover_cmds_hifi)
hifi_dyn.set_initial_conditions(
    INITIAL_POSITION,
    INITIAL_VELOCITY,
    INITIAL_ATTITUDE,
    INITIAL_ANGULAR_VELOCITY,
    REF_LAT,
    REF_LON,
    TERRAIN_ALT,
    ned_mag,
)
hifi_dyn.vehicle.takenoff = True

# Simulation parameters for Test 2
SIM_TIME_2 = 10.0  # seconds
num_steps_2 = int(SIM_TIME_2 / DT)

# Roll maneuver parameters
ROLL_TIME = 8.0  # Time to complete 360 degree roll
ROLL_RATE = 2.0 * np.pi / ROLL_TIME  # rad/s (constant roll rate)

print(f"Roll maneuver time: {ROLL_TIME} s")
print(f"Roll rate: {np.rad2deg(ROLL_RATE):.2f} deg/s")
print(f"Simulation time: {SIM_TIME_2} s")

# Storage
time_hist_2 = np.zeros(num_steps_2)
body_vel_hist_2 = np.zeros((num_steps_2, 3))
body_vel_ref_hist_2 = np.zeros((num_steps_2, 3))
body_omega_hist_2 = np.zeros((num_steps_2, 3))
body_omega_ref_hist_2 = np.zeros((num_steps_2, 3))
ned_pos_hist_2 = np.zeros((num_steps_2, 3))
euler_hist_2 = np.zeros((num_steps_2, 3))
cmd_hist_2 = np.zeros((num_steps_2, num_motors))

# Initial conditions
cur_state = hifi_dyn.vehicle.state.copy()
cur_input = hover_cmds_hifi.copy()

# Initialize low-pass filter states for sensor measurements
vel_filt = np.zeros(3)
accel_filt = np.zeros(3)
omega_filt = np.zeros(3)
alpha_filt = np.zeros(3)

# Previous omega measurement for numerical differentiation
omega_prev = np.zeros(3)

print("Running Test 2 simulation with wind...")
for ii in range(num_steps_2):
    tt = ii * DT
    time_hist_2[ii] = tt

    # Current true state
    body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
    body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
    body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
    quat = cur_state[v_smap_quat.quat].flatten()
    ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

    # Compute wind with gusts
    wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
    wind_total = WIND_VELOCITY + wind_gust

    # Compute wind drag force in body frame
    wind_force_body = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)

    # Realistic sensor model (same as Test 1)
    vel_noise = np.random.normal(0, SIGMA_VEL, 3)
    body_vel_meas = body_vel_true + vel_noise + BIAS_VEL
    vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt

    accel_noise = np.random.normal(0, SIGMA_ACCEL, 3)
    body_accel_meas = body_accel_true + accel_noise + BIAS_ACCEL
    accel_filt = alpha_accel * body_accel_meas + (1.0 - alpha_accel) * accel_filt

    omega_noise = np.random.normal(0, SIGMA_OMEGA, 3)
    body_omega_meas = body_omega_true + omega_noise + BIAS_OMEGA
    omega_filt = alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt

    if ii == 0:
        alpha_meas = np.zeros(3)
    else:
        alpha_meas = (omega_filt - omega_prev) / DT
    alpha_filt = alpha_alpha * alpha_meas + (1.0 - alpha_alpha) * alpha_filt
    omega_prev = omega_filt.copy()

    # State and derivatives for INDI controller
    x = np.concatenate([vel_filt, omega_filt])
    x_dot = np.concatenate([accel_filt, alpha_filt])

    # Reference: zero velocity, constant roll rate during maneuver
    vb_ref = np.zeros(3)
    if tt < ROLL_TIME:
        omega_ref = np.array([ROLL_RATE, 0.0, 0.0])
    else:
        omega_ref = np.zeros(3)
    ref = np.concatenate([vb_ref, omega_ref])

    # Reference derivative (all zeros for this test)
    ref_dot = np.zeros(6)

    # INDI control
    u_cmd = indi_ctrl.calculate_control(
        cur_time=tt,
        cur_state=x,
        cur_state_dot=x_dot,
        cur_input=cur_input,
        ref=ref,
        ref_dot=ref_dot,
    )
    u_cmd = np.clip(u_cmd, -1.0, 1.0)

    # Store data
    body_vel_hist_2[ii, :] = body_vel_true
    body_vel_ref_hist_2[ii, :] = vb_ref
    body_omega_hist_2[ii, :] = body_omega_true
    body_omega_ref_hist_2[ii, :] = omega_ref
    ned_pos_hist_2[ii, :] = cur_state[v_smap_quat.ned_pos].flatten()
    cmd_hist_2[ii, :] = u_cmd

    # Convert quaternion to Euler angles for plotting
    roll, pitch, yaw = gmath.quat_to_euler(quat)
    euler_hist_2[ii, :] = np.rad2deg([roll, pitch, yaw])

    # Propagate dynamics with wind disturbance
    next_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()

    # Add wind force as external acceleration
    wind_accel_body = wind_force_body / mass
    next_state[v_smap_quat.body_vel] += wind_accel_body * DT

    cur_state = next_state
    cur_input = motor_effector.state.copy()

print("Test 2 complete!")
print(f"Final velocity: {np.linalg.norm(body_vel_hist_2[-1, :]):.4f} m/s")
print(f"Final roll angle: {euler_hist_2[-1, 0]:.2f} deg")
print(f"Final angular rate: {np.linalg.norm(body_omega_hist_2[-1, :]):.6f} rad/s\n")


# ============================================================================
# Plot Test 2 results
# ============================================================================
print("Plotting Test 2 results...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig2.suptitle(
    "Test 2: 360 Degree Roll Maneuver (Zero Velocity)",
    fontsize=14,
    fontweight="bold",
)

# Body velocity
ax = axes2[0, 0]
ax.plot(time_hist_2, body_vel_hist_2[:, 0], "r-", label="vb_x")
ax.plot(time_hist_2, body_vel_hist_2[:, 1], "g-", label="vb_y")
ax.plot(time_hist_2, body_vel_hist_2[:, 2], "b-", label="vb_z")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Velocity (m/s)")
ax.set_ylim([-1.3, 1.3])
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Body angular rate
ax = axes2[0, 1]
ax.plot(time_hist_2, np.rad2deg(body_omega_hist_2[:, 0]), "r-", label="p")
ax.plot(
    time_hist_2,
    np.rad2deg(body_omega_ref_hist_2[:, 0]),
    "r--",
    alpha=0.7,
    label="p ref",
)
ax.plot(time_hist_2, np.rad2deg(body_omega_hist_2[:, 1]), "g-", label="q")
ax.plot(time_hist_2, np.rad2deg(body_omega_hist_2[:, 2]), "b-", label="r")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Angular Rate (deg/s)")
ax.set_ylim([-65, 65])
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# Euler angles (roll should go from 0 to 360)
ax = axes2[1, 0]
ax.plot(time_hist_2, euler_hist_2[:, 0], "r-", label="Roll", linewidth=2)
ax.plot(time_hist_2, euler_hist_2[:, 1], "g-", label="Pitch")
ax.plot(time_hist_2, euler_hist_2[:, 2], "b-", label="Yaw")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Euler Angles (deg)")
ax.set_ylim([-180, 180])
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Motor commands
ax = axes2[1, 1]
for ii in range(num_motors):
    ax.plot(time_hist_2, cmd_hist_2[:, ii], alpha=0.7, label=f"u{ii}")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Motor Commands")
ax.set_ylim([-1.0, 1.0])
ax.legend(loc="upper right", ncol=4, fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_file_2 = output_dir / "indi_test2_360_roll_wind_copy.png"
plt.savefig(output_file_2, dpi=150)
print(f"Test 2 figure saved to: {output_file_2}")


# ============================================================================
# Test 3: Combined velocity + attitude maneuver (with wind)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: Combined Velocity + Attitude Maneuver (with wind)")
print("=" * 70)

# Reset dynamics to hover
motor_effector.set_initial_state(hover_cmds_hifi)
hifi_dyn.set_initial_conditions(
    INITIAL_POSITION,
    INITIAL_VELOCITY,
    INITIAL_ATTITUDE,
    INITIAL_ANGULAR_VELOCITY,
    REF_LAT,
    REF_LON,
    TERRAIN_ALT,
    ned_mag,
)
hifi_dyn.vehicle.takenoff = True

# Simulation parameters for Test 3
SIM_TIME_3 = 12.0  # seconds
num_steps_3 = int(SIM_TIME_3 / DT)

# Combined maneuver parameters
V_CIRCLE_3 = 0.8  # m/s (slightly slower for combined maneuver)
OMEGA_CIRCLE_3 = 2.0 * np.pi / 10.0  # rad/s (10 second period)
ROLL_TIME_3 = 8.0  # Time to complete roll
ROLL_RATE_3 = 2.0 * np.pi / ROLL_TIME_3  # rad/s

print(f"Circle velocity: {V_CIRCLE_3} m/s")
print(f"Circle period: {2*np.pi/OMEGA_CIRCLE_3:.2f} s")
print(f"Roll maneuver time: {ROLL_TIME_3} s")
print(f"Roll rate: {np.rad2deg(ROLL_RATE_3):.2f} deg/s")
print(f"Simulation time: {SIM_TIME_3} s")

# Storage
time_hist_3 = np.zeros(num_steps_3)
body_vel_hist_3 = np.zeros((num_steps_3, 3))
body_vel_ref_hist_3 = np.zeros((num_steps_3, 3))
body_omega_hist_3 = np.zeros((num_steps_3, 3))
body_omega_ref_hist_3 = np.zeros((num_steps_3, 3))
ned_pos_hist_3 = np.zeros((num_steps_3, 3))
euler_hist_3 = np.zeros((num_steps_3, 3))
cmd_hist_3 = np.zeros((num_steps_3, num_motors))

# Initial conditions
cur_state = hifi_dyn.vehicle.state.copy()
cur_input = hover_cmds_hifi.copy()

# Initialize low-pass filter states for sensor measurements
vel_filt = np.zeros(3)
accel_filt = np.zeros(3)
omega_filt = np.zeros(3)
alpha_filt = np.zeros(3)

# Previous omega measurement for numerical differentiation
omega_prev = np.zeros(3)

print("Running Test 3 simulation with wind...")
for ii in range(num_steps_3):
    tt = ii * DT
    time_hist_3[ii] = tt

    # Current true state
    body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
    body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
    body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
    quat = cur_state[v_smap_quat.quat].flatten()
    ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

    # Compute wind with gusts
    wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
    wind_total = WIND_VELOCITY + wind_gust

    # Compute wind drag force in body frame
    wind_force_body = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)

    # Realistic sensor model (same as Test 1 and 2)
    vel_noise = np.random.normal(0, SIGMA_VEL, 3)
    body_vel_meas = body_vel_true + vel_noise + BIAS_VEL
    vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt

    accel_noise = np.random.normal(0, SIGMA_ACCEL, 3)
    body_accel_meas = body_accel_true + accel_noise + BIAS_ACCEL
    accel_filt = alpha_accel * body_accel_meas + (1.0 - alpha_accel) * accel_filt

    omega_noise = np.random.normal(0, SIGMA_OMEGA, 3)
    body_omega_meas = body_omega_true + omega_noise + BIAS_OMEGA
    omega_filt = alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt

    if ii == 0:
        alpha_meas = np.zeros(3)
    else:
        alpha_meas = (omega_filt - omega_prev) / DT
    alpha_filt = alpha_alpha * alpha_meas + (1.0 - alpha_alpha) * alpha_filt
    omega_prev = omega_filt.copy()

    # State and derivatives for INDI controller
    x = np.concatenate([vel_filt, omega_filt])
    x_dot = np.concatenate([accel_filt, alpha_filt])

    # Reference: circular velocity + roll maneuver (both active simultaneously)
    vb_ref = np.array(
        [
            V_CIRCLE_3 * np.cos(OMEGA_CIRCLE_3 * tt),
            V_CIRCLE_3 * np.sin(OMEGA_CIRCLE_3 * tt),
            0.0,
        ]
    )

    if tt < ROLL_TIME_3:
        omega_ref = np.array([ROLL_RATE_3, 0.0, 0.0])
    else:
        omega_ref = np.zeros(3)

    ref = np.concatenate([vb_ref, omega_ref])

    # Reference derivative
    vb_ref_dot = np.array(
        [
            -V_CIRCLE_3 * OMEGA_CIRCLE_3 * np.sin(OMEGA_CIRCLE_3 * tt),
            V_CIRCLE_3 * OMEGA_CIRCLE_3 * np.cos(OMEGA_CIRCLE_3 * tt),
            0.0,
        ]
    )
    omega_ref_dot = np.zeros(3)
    ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

    # INDI control
    u_cmd = indi_ctrl.calculate_control(
        cur_time=tt,
        cur_state=x,
        cur_state_dot=x_dot,
        cur_input=cur_input,
        ref=ref,
        ref_dot=ref_dot,
    )
    u_cmd = np.clip(u_cmd, -1.0, 1.0)

    # Store data
    body_vel_hist_3[ii, :] = body_vel_true
    body_vel_ref_hist_3[ii, :] = vb_ref
    body_omega_hist_3[ii, :] = body_omega_true
    body_omega_ref_hist_3[ii, :] = omega_ref
    ned_pos_hist_3[ii, :] = cur_state[v_smap_quat.ned_pos].flatten()
    cmd_hist_3[ii, :] = u_cmd

    # Convert quaternion to Euler angles for plotting
    roll, pitch, yaw = gmath.quat_to_euler(quat)
    euler_hist_3[ii, :] = np.rad2deg([roll, pitch, yaw])

    # Propagate dynamics with wind disturbance
    next_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()

    # Add wind force as external acceleration
    wind_accel_body = wind_force_body / mass
    next_state[v_smap_quat.body_vel] += wind_accel_body * DT

    cur_state = next_state
    cur_input = motor_effector.state.copy()

print("Test 3 complete!")
print(
    f"Final velocity error: {np.linalg.norm(body_vel_hist_3[-1, :] - body_vel_ref_hist_3[-1, :]):.4f} m/s"
)
print(f"Final roll angle: {euler_hist_3[-1, 0]:.2f} deg")
print(f"Final angular rate: {np.linalg.norm(body_omega_hist_3[-1, :]):.6f} rad/s\n")


# ============================================================================
# Plot Test 3 results
# ============================================================================
print("Plotting Test 3 results...")

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig3.suptitle(
    "Test 3: Combined Velocity + Attitude Maneuver (with wind)",
    fontsize=14,
    fontweight="bold",
)

# Body velocity tracking
ax = axes3[0, 0]
ax.plot(time_hist_3, body_vel_hist_3[:, 0], "r-", label="vb_x")
ax.plot(time_hist_3, body_vel_ref_hist_3[:, 0], "r--", alpha=0.7, label="vb_x ref")
ax.plot(time_hist_3, body_vel_hist_3[:, 1], "g-", label="vb_y")
ax.plot(time_hist_3, body_vel_ref_hist_3[:, 1], "g--", alpha=0.7, label="vb_y ref")
ax.plot(time_hist_3, body_vel_hist_3[:, 2], "b-", label="vb_z")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Velocity (m/s)")
ax.set_ylim([-1.3, 1.3])
ax.legend(loc="upper right", ncol=3, fontsize=8)
ax.grid(True, alpha=0.3)

# Body angular rate
ax = axes3[0, 1]
ax.plot(time_hist_3, np.rad2deg(body_omega_hist_3[:, 0]), "r-", label="p")
ax.plot(
    time_hist_3,
    np.rad2deg(body_omega_ref_hist_3[:, 0]),
    "r--",
    alpha=0.7,
    label="p ref",
)
ax.plot(time_hist_3, np.rad2deg(body_omega_hist_3[:, 1]), "g-", label="q")
ax.plot(time_hist_3, np.rad2deg(body_omega_hist_3[:, 2]), "b-", label="r")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Angular Rate (deg/s)")
ax.set_ylim([-65, 65])
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# Euler angles
ax = axes3[1, 0]
ax.plot(time_hist_3, euler_hist_3[:, 0], "r-", label="Roll", linewidth=2)
ax.plot(time_hist_3, euler_hist_3[:, 1], "g-", label="Pitch")
ax.plot(time_hist_3, euler_hist_3[:, 2], "b-", label="Yaw")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Euler Angles (deg)")
ax.set_ylim([-180, 180])
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Motor commands
ax = axes3[1, 1]
for ii in range(num_motors):
    ax.plot(time_hist_3, cmd_hist_3[:, ii], alpha=0.7, label=f"u{ii}")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Motor Commands")
ax.set_ylim([-1.0, 1.0])
ax.legend(loc="upper right", ncol=4, fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_file_3 = output_dir / "indi_test3_combined_wind_copy.png"
plt.savefig(output_file_3, dpi=150)
print(f"Test 3 figure saved to: {output_file_3}")


# ============================================================================
# Test 4: Helix trajectory with 3-axis tumble (with wind)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: Helix Trajectory with 3-Axis Tumble (with wind)")
print("=" * 70)

# Reset dynamics to hover
motor_effector.set_initial_state(hover_cmds_hifi)
hifi_dyn.set_initial_conditions(
    INITIAL_POSITION,
    INITIAL_VELOCITY,
    INITIAL_ATTITUDE,
    INITIAL_ANGULAR_VELOCITY,
    REF_LAT,
    REF_LON,
    TERRAIN_ALT,
    ned_mag,
)
hifi_dyn.vehicle.takenoff = True

# Simulation parameters for Test 4
SIM_TIME_4 = 12.0  # seconds
num_steps_4 = int(SIM_TIME_4 / DT)

# Helix trajectory parameters
V_CIRCLE_4 = 0.6  # m/s horizontal circle velocity
OMEGA_CIRCLE_4 = 2.0 * np.pi / 10.0  # rad/s (10 second period for horizontal circle)
V_VERT_AMP = 0.3  # m/s vertical velocity amplitude
OMEGA_VERT = 2.0 * np.pi / 6.0  # rad/s (6 second period for vertical oscillation)

# Tumble parameters (all axes active - full rotations on all axes!)
ROLL_RATE_4 = 2.0 * np.pi / 8.0  # rad/s (360degrees roll in 8 seconds)
PITCH_RATE_4 = 2.0 * np.pi / 10.0  # rad/s (360degrees pitch flip in 10 seconds)
TUMBLE_TIME = 10.0  # Time for tumble maneuver
YAW_RATE_4 = 2.0 * np.pi / 12.0  # rad/s (360degrees yaw in 12 seconds)

print(
    f"Helix - Horizontal circle: {V_CIRCLE_4} m/s, period: {2*np.pi/OMEGA_CIRCLE_4:.1f} s"
)
print(
    f"Helix - Vertical oscillation: ±{V_VERT_AMP} m/s, period: {2*np.pi/OMEGA_VERT:.1f} s"
)
print(f"Tumble - Roll rate: {np.rad2deg(ROLL_RATE_4):.1f} deg/s")
print(f"Tumble - Pitch rate: {np.rad2deg(PITCH_RATE_4):.1f} deg/s")
print(f"Tumble - Yaw rate: {np.rad2deg(YAW_RATE_4):.1f} deg/s")
print(f"Simulation time: {SIM_TIME_4} s")

# Storage
time_hist_4 = np.zeros(num_steps_4)
body_vel_hist_4 = np.zeros((num_steps_4, 3))
body_vel_ref_hist_4 = np.zeros((num_steps_4, 3))
body_omega_hist_4 = np.zeros((num_steps_4, 3))
body_omega_ref_hist_4 = np.zeros((num_steps_4, 3))
ned_pos_hist_4 = np.zeros((num_steps_4, 3))
euler_hist_4 = np.zeros((num_steps_4, 3))
cmd_hist_4 = np.zeros((num_steps_4, num_motors))

# Initial conditions
cur_state = hifi_dyn.vehicle.state.copy()
cur_input = hover_cmds_hifi.copy()

# Initialize low-pass filter states for sensor measurements
vel_filt = np.zeros(3)
accel_filt = np.zeros(3)
omega_filt = np.zeros(3)
alpha_filt = np.zeros(3)

# Previous omega measurement for numerical differentiation
omega_prev = np.zeros(3)

print("Running Test 4 simulation with wind...")
for ii in range(num_steps_4):
    tt = ii * DT
    time_hist_4[ii] = tt

    # Current true state
    body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
    body_accel_true = cur_state[v_smap_quat.body_accel].flatten()
    body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
    quat = cur_state[v_smap_quat.quat].flatten()
    ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

    # Compute wind with gusts
    wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
    wind_total = WIND_VELOCITY + wind_gust

    # Compute wind drag force in body frame
    wind_force_body = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)

    # Realistic sensor model (same as other tests)
    vel_noise = np.random.normal(0, SIGMA_VEL, 3)
    body_vel_meas = body_vel_true + vel_noise + BIAS_VEL
    vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt

    accel_noise = np.random.normal(0, SIGMA_ACCEL, 3)
    body_accel_meas = body_accel_true + accel_noise + BIAS_ACCEL
    accel_filt = alpha_accel * body_accel_meas + (1.0 - alpha_accel) * accel_filt

    omega_noise = np.random.normal(0, SIGMA_OMEGA, 3)
    body_omega_meas = body_omega_true + omega_noise + BIAS_OMEGA
    omega_filt = alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt

    if ii == 0:
        alpha_meas = np.zeros(3)
    else:
        alpha_meas = (omega_filt - omega_prev) / DT
    alpha_filt = alpha_alpha * alpha_meas + (1.0 - alpha_alpha) * alpha_filt
    omega_prev = omega_filt.copy()

    # State and derivatives for INDI controller
    x = np.concatenate([vel_filt, omega_filt])
    x_dot = np.concatenate([accel_filt, alpha_filt])

    # Helix velocity reference: horizontal circle + vertical oscillation
    vb_ref = np.array(
        [
            V_CIRCLE_4 * np.cos(OMEGA_CIRCLE_4 * tt),
            V_CIRCLE_4 * np.sin(OMEGA_CIRCLE_4 * tt),
            V_VERT_AMP * np.sin(OMEGA_VERT * tt),
        ]
    )

    # Tumble angular rate reference: constant rates on all axes
    if tt < TUMBLE_TIME:
        omega_ref = np.array([ROLL_RATE_4, PITCH_RATE_4, YAW_RATE_4])
    else:
        omega_ref = np.array([0.0, 0.0, 0.0])  # Stop tumble after some time

    ref = np.concatenate([vb_ref, omega_ref])

    # Reference derivatives
    vb_ref_dot = np.array(
        [
            -V_CIRCLE_4 * OMEGA_CIRCLE_4 * np.sin(OMEGA_CIRCLE_4 * tt),
            V_CIRCLE_4 * OMEGA_CIRCLE_4 * np.cos(OMEGA_CIRCLE_4 * tt),
            V_VERT_AMP * OMEGA_VERT * np.cos(OMEGA_VERT * tt),
        ]
    )
    omega_ref_dot = np.zeros(3)  # Constant rates have zero derivative
    ref_dot = np.concatenate([vb_ref_dot, omega_ref_dot])

    # INDI control
    u_cmd = indi_ctrl.calculate_control(
        cur_time=tt,
        cur_state=x,
        cur_state_dot=x_dot,
        cur_input=cur_input,
        ref=ref,
        ref_dot=ref_dot,
    )
    u_cmd = np.clip(u_cmd, -1.0, 1.0)

    # Store data
    body_vel_hist_4[ii, :] = body_vel_true
    body_vel_ref_hist_4[ii, :] = vb_ref
    body_omega_hist_4[ii, :] = body_omega_true
    body_omega_ref_hist_4[ii, :] = omega_ref
    ned_pos_hist_4[ii, :] = cur_state[v_smap_quat.ned_pos].flatten()
    cmd_hist_4[ii, :] = u_cmd

    # Convert quaternion to Euler angles for plotting
    roll, pitch, yaw = gmath.quat_to_euler(quat)
    euler_hist_4[ii, :] = np.rad2deg([roll, pitch, yaw])

    # Propagate dynamics with wind disturbance
    next_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()

    # Add wind force as external acceleration
    wind_accel_body = wind_force_body / mass
    next_state[v_smap_quat.body_vel] += wind_accel_body * DT

    cur_state = next_state
    cur_input = motor_effector.state.copy()

print("Test 4 complete!")
print(
    f"Final velocity error: {np.linalg.norm(body_vel_hist_4[-1, :] - body_vel_ref_hist_4[-1, :]):.4f} m/s"
)
print(f"Final altitude change: {ned_pos_hist_4[-1, 2] - ned_pos_hist_4[0, 2]:.2f} m")
print(
    f"Max altitude change: {(ned_pos_hist_4[:, 2] - ned_pos_hist_4[0, 2]).max():.2f} m"
)
print(f"Final angular rate: {np.linalg.norm(body_omega_hist_4[-1, :]):.6f} rad/s\n")


# ============================================================================
# Plot Test 4 results
# ============================================================================
print("Plotting Test 4 results...")

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig4.suptitle(
    "Test 4: Helix Trajectory with 3-Axis Tumble (with wind)",
    fontsize=14,
    fontweight="bold",
)

# Body velocity tracking (helix)
ax = axes4[0, 0]
ax.plot(time_hist_4, body_vel_hist_4[:, 0], "r-", label="vb_x")
ax.plot(time_hist_4, body_vel_ref_hist_4[:, 0], "r--", alpha=0.7, label="vb_x ref")
ax.plot(time_hist_4, body_vel_hist_4[:, 1], "g-", label="vb_y")
ax.plot(time_hist_4, body_vel_ref_hist_4[:, 1], "g--", alpha=0.7, label="vb_y ref")
ax.plot(time_hist_4, body_vel_hist_4[:, 2], "b-", label="vb_z")
ax.plot(time_hist_4, body_vel_ref_hist_4[:, 2], "b--", alpha=0.7, label="vb_z ref")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Velocity (m/s)")
ax.set_ylim([-1.3, 1.3])
ax.legend(loc="upper right", ncol=3, fontsize=7)
ax.grid(True, alpha=0.3)

# Body angular rate (tumble on all axes)
ax = axes4[0, 1]
ax.plot(time_hist_4, np.rad2deg(body_omega_hist_4[:, 0]), "r-", label="p")
ax.plot(
    time_hist_4,
    np.rad2deg(body_omega_ref_hist_4[:, 0]),
    "r--",
    alpha=0.5,
    label="p ref",
)
ax.plot(time_hist_4, np.rad2deg(body_omega_hist_4[:, 1]), "g-", label="q")
ax.plot(
    time_hist_4,
    np.rad2deg(body_omega_ref_hist_4[:, 1]),
    "g--",
    alpha=0.5,
    label="q ref",
)
ax.plot(time_hist_4, np.rad2deg(body_omega_hist_4[:, 2]), "b-", label="r")
ax.plot(
    time_hist_4,
    np.rad2deg(body_omega_ref_hist_4[:, 2]),
    "b--",
    alpha=0.5,
    label="r ref",
)
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Angular Rate (deg/s)")
ax.set_ylim([-65, 65])
ax.legend(loc="upper right", ncol=3, fontsize=7)
ax.grid(True, alpha=0.3)

# Euler angles (tumbling on all axes)
ax = axes4[1, 0]
ax.plot(time_hist_4, euler_hist_4[:, 0], "r-", label="Roll", linewidth=1.5)
ax.plot(time_hist_4, euler_hist_4[:, 1], "g-", label="Pitch", linewidth=1.5)
ax.plot(time_hist_4, euler_hist_4[:, 2], "b-", label="Yaw", linewidth=1.5)
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Euler Angles (deg)")
ax.set_ylim([-180, 180])
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Motor commands
ax = axes4[1, 1]
for ii in range(num_motors):
    ax.plot(time_hist_4, cmd_hist_4[:, ii], alpha=0.7, label=f"u{ii}")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Motor Commands")
ax.set_ylim([-1.0, 1.0])
ax.legend(loc="upper right", ncol=4, fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_file_4 = output_dir / "indi_test4_helix_tumble_wind_copy.png"
plt.savefig(output_file_4, dpi=150)
print(f"Test 4 figure saved to: {output_file_4}")


plt.show()
