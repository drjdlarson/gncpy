"""Test omnicopter INDI controller with various maneuvers.

This script tests INDI control with:
    Test 1: Circular velocity tracking (zero angular rates)
    Test 2: 360 degree roll maneuver (zero velocity)
    Test 3: Velocity + attitude maneuver
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
# First-order motor dynamics effector
# ============================================================================
class MotorDynamicsEffector(Effector):
    """First-order motor dynamics model.

    Models motor response as a first-order lag:
        omegȧ_i = (1/tau_mot) * (omega_cmd,i - omega_i)
    """

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
        """Propagate motor dynamics one timestep using exact solution."""
        input_cmds = np.array(input_cmds).flatten()
        alpha = np.exp(-dt / self.tau_mot)
        self.state = input_cmds + (self.state - input_cmds) * alpha
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

# Measurement noise (set to 0 for perfect measurements)
SIGMA_VEL = 0.05  # m/s (IMU-derived velocity noise)
SIGMA_OMEGA = 0.01  # rad/s (gyroscope noise)

# Optional measurement bias (set to 0 for no bias)
BIAS_VEL = np.array([0.0, 0.0, 0.0])  # m/s
BIAS_OMEGA = np.array([0.0, 0.0, 0.0])  # rad/s

# Low-pass filter for measurements (reduces noise amplification in INDI)
FC_VEL = 5.0  # Hz (velocity filter cutoff)
FC_OMEGA = 10.0  # Hz (angular rate filter cutoff)

# Wind parameters (NED frame) - increased significantly to show wind rejection
WIND_VELOCITY = np.array([5.0, 3.0, 0.5])  # m/s (constant wind in NED)
WIND_GUST_AMP = np.array([2.0, 1.5, 0.5])  # m/s (gust amplitude)
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
    effective_area = frontal_area[:3] * 10.0

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
hifi_config_file = Path(__file__).parent / "omnicopter_config_hifi.yaml"

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

# Initialize low-pass filter states
vel_filt = np.zeros(3)
omega_filt = np.zeros(3)
alpha_vel = DT / (DT + 1.0 / (2.0 * np.pi * FC_VEL))
alpha_omega = DT / (DT + 1.0 / (2.0 * np.pi * FC_OMEGA))

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

    # Current state (true)
    body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
    body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
    quat = cur_state[v_smap_quat.quat].flatten()
    ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

    # Compute wind with gusts
    wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
    wind_total = WIND_VELOCITY + wind_gust

    # Compute wind drag force in body frame
    wind_force_body = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)

    # Add measurement noise and bias
    vel_noise = np.random.normal(0, SIGMA_VEL, 3)
    omega_noise = np.random.normal(0, SIGMA_OMEGA, 3)
    body_vel_meas = body_vel_true + vel_noise + BIAS_VEL
    body_omega_meas = body_omega_true + omega_noise + BIAS_OMEGA

    # Low-pass filter measurements (first-order discrete filter)
    vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt
    omega_filt = alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt

    # Filtered state for controller
    x = np.concatenate([vel_filt, omega_filt])

    # State derivatives
    body_accel = cur_state[v_smap_quat.body_accel].flatten()
    body_rot_accel = cur_state[v_smap_quat.body_rot_accel].flatten()
    x_dot = np.concatenate([body_accel, body_rot_accel])

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
ax.set_ylim([-1.0, 1.0])
ax.legend(loc="upper right", ncol=3, fontsize=8)
ax.grid(True, alpha=0.3)

# Body angular rate
ax = axes1[0, 1]
ax.plot(time_hist_1, np.rad2deg(body_omega_hist_1[:, 0]), "r-", label="p")
ax.plot(time_hist_1, np.rad2deg(body_omega_hist_1[:, 1]), "g-", label="q")
ax.plot(time_hist_1, np.rad2deg(body_omega_hist_1[:, 2]), "b-", label="r")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Angular Rate (deg/s)")
ax.set_ylim([-50, 50])
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Euler angles
ax = axes1[1, 0]
ax.plot(time_hist_1, euler_hist_1[:, 0], "r-", label="Roll")
ax.plot(time_hist_1, euler_hist_1[:, 1], "g-", label="Pitch")
ax.plot(time_hist_1, euler_hist_1[:, 2], "b-", label="Yaw")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Euler Angles (deg)")
ax.set_ylim([-5, 5])
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
output_file_1 = output_dir / "indi_test1_circular_velocity_wind.png"
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

# Initialize low-pass filter states
vel_filt = np.zeros(3)
omega_filt = np.zeros(3)

print("Running Test 2 simulation with wind...")
for ii in range(num_steps_2):
    tt = ii * DT
    time_hist_2[ii] = tt

    # Current state (true)
    body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
    body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
    quat = cur_state[v_smap_quat.quat].flatten()
    ned_vel = cur_state[v_smap_quat.ned_vel].flatten()

    # Compute wind with gusts
    wind_gust = WIND_GUST_AMP * np.sin(2.0 * np.pi * WIND_GUST_FREQ * tt)
    wind_total = WIND_VELOCITY + wind_gust

    # Compute wind drag force in body frame
    wind_force_body = compute_wind_force(wind_total, ned_vel, quat, cd, frontal_area)

    # Add measurement noise and bias
    vel_noise = np.random.normal(0, SIGMA_VEL, 3)
    omega_noise = np.random.normal(0, SIGMA_OMEGA, 3)
    body_vel_meas = body_vel_true + vel_noise + BIAS_VEL
    body_omega_meas = body_omega_true + omega_noise + BIAS_OMEGA

    # Low-pass filter measurements
    vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt
    omega_filt = alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt

    # Filtered state for controller
    x = np.concatenate([vel_filt, omega_filt])

    # State derivatives
    body_accel = cur_state[v_smap_quat.body_accel].flatten()
    body_rot_accel = cur_state[v_smap_quat.body_rot_accel].flatten()
    x_dot = np.concatenate([body_accel, body_rot_accel])

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
ax.set_ylim([-1.0, 1.0])
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
ax.set_ylim([-50, 50])
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# Euler angles (roll should go from 0 to 360)
ax = axes2[1, 0]
ax.plot(time_hist_2, euler_hist_2[:, 0], "r-", label="Roll", linewidth=2)
ax.plot(time_hist_2, euler_hist_2[:, 1], "g-", label="Pitch")
ax.plot(time_hist_2, euler_hist_2[:, 2], "b-", label="Yaw")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Euler Angles (deg)")
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

output_file_2 = output_dir / "indi_test2_360_roll_wind.png"
plt.savefig(output_file_2, dpi=150)
print(f"Test 2 figure saved to: {output_file_2}")


# ============================================================================
# Test 3: Combined velocity + attitude maneuver
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: Combined Velocity + Attitude Maneuver")
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

# Initialize low-pass filter states
vel_filt = np.zeros(3)
omega_filt = np.zeros(3)

print("Running Test 3 simulation...")
for ii in range(num_steps_3):
    tt = ii * DT
    time_hist_3[ii] = tt

    # Current state (true)
    body_vel_true = cur_state[v_smap_quat.body_vel].flatten()
    body_omega_true = cur_state[v_smap_quat.body_rot_rate].flatten()
    quat = cur_state[v_smap_quat.quat].flatten()

    # Add measurement noise and bias
    vel_noise = np.random.normal(0, SIGMA_VEL, 3)
    omega_noise = np.random.normal(0, SIGMA_OMEGA, 3)
    body_vel_meas = body_vel_true + vel_noise + BIAS_VEL
    body_omega_meas = body_omega_true + omega_noise + BIAS_OMEGA

    # Low-pass filter measurements
    vel_filt = alpha_vel * body_vel_meas + (1.0 - alpha_vel) * vel_filt
    omega_filt = alpha_omega * body_omega_meas + (1.0 - alpha_omega) * omega_filt

    # Filtered state for controller
    x = np.concatenate([vel_filt, omega_filt])

    # State derivatives
    body_accel = cur_state[v_smap_quat.body_accel].flatten()
    body_rot_accel = cur_state[v_smap_quat.body_rot_accel].flatten()
    x_dot = np.concatenate([body_accel, body_rot_accel])

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

    # Propagate dynamics
    cur_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()
    cur_input = motor_effector.state.copy()

print("Test 3 complete!")
print(
    f"Final velocity error: {np.linalg.norm(body_vel_hist_3[-1, :] - body_vel_ref_hist_3[-1, :]):.4f} m/s"
)
print(f"Final roll angle: {euler_hist_3[-1, 0]:.2f} deg")
print(f"Final angular rate: {np.linalg.norm(body_omega_hist_3[-1, :]):.6f} rad/s\n")


plt.show()
