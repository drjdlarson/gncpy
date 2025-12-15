"""Test omnicopter with INDI controller. Te test start off
with a ton of validation of the physics. first determining the correct hover thrust
besed on the low fidelity model, makes the B matrix for INDi and uses it to compute
the hover thrust, and then we verify the motor tunr directions after we get the hover thrust.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
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
        Ω̇_i = (1/τ_mot) * (Ω_cmd,i - Ω_i)

    This gives exponential convergence of motor speed to commanded speed
    with time constant τ_mot.

    Parameters
    ----------
    num_motors : int
        Number of motors.
    tau_mot : float
        Motor time constant in seconds.
    initial_state : numpy array, optional
        Initial motor commands/speeds. Defaults to zeros.
    """

    def __init__(self, num_motors, tau_mot, initial_state=None):
        self.num_motors = num_motors
        self.tau_mot = tau_mot

        # Motor state (current "speed" or command level)
        if initial_state is not None:
            self.state = np.array(initial_state).flatten().copy()
        else:
            self.state = np.zeros(num_motors)

    def set_initial_state(self, initial_state):
        """Set the initial motor state.

        Parameters
        ----------
        initial_state : numpy array
            Initial motor commands/speeds.
        """
        self.state = np.array(initial_state).flatten().copy()

    def step(self, input_cmds, dt):
        """Propagate motor dynamics one timestep.

        Uses exact solution of first-order ODE:
            Ω(t+dt) = Ω_cmd + (Ω(t) - Ω_cmd) * exp(-dt/τ)

        Parameters
        ----------
        input_cmds : numpy array
            Commanded motor speeds/throttles.
        dt : float
            Timestep in seconds.

        Returns
        -------
        numpy array
            Current motor state after update.
        """
        input_cmds = np.array(input_cmds).flatten()

        # Exact solution of first-order lag
        alpha = np.exp(-dt / self.tau_mot)
        self.state = input_cmds + (self.state - input_cmds) * alpha

        return self.state.copy()


# Define costants
DT = 0.001  # Timestep
SIM_TIME = 10.0  # Total simulation time

# note we are using NED global values and body frame with z down as well when there is zero rpy
INITIAL_POSITION = np.array([0.0, 0.0, -10.0])  # Initial position (x, y, z)
INITIAL_VELOCITY = np.array([0.0, 0.0, 0.0])  # Initial velocity (vx, vy, vz)
INITIAL_ATTITUDE = np.array([0.0, 0.0, 0.0])  # Initial attitude (roll, pitch, yaw)
INITIAL_ANGULAR_VELOCITY = np.array(
    [0.0, 0.0, 0.0]
)  # Initial angular velocity (p, q, r)

REF_LAT, REF_LON, TERRAIN_ALT = 34.0, -86.0, 0.0
TARGET_POS = np.array([2.0, 1.0, -2.5])  # NED (m)

GRAVITY = 9.81  # m/s^2


# ============================================================================
# Load Low Fidelity Omnicopter Dynamics (no motor dynamics - for INDI design)
# ============================================================================
print("Loading low fidelity omnicopter...")
lofi_config_file = Path(__file__).parent / "omnicopter_config.yaml"
hifi_config_file = Path(__file__).parent / "omnicopter_config_hifi.yaml"

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

# Print gravity vector from environment
gravity = lofi_dyn.env.state[e_smap.gravity]
print(f"Gravity vector (NED): {gravity}")

# ============================================================================
# Build B0 matrix for INDI
# ============================================================================
# B0 = [ (1/m) * T_max * N        ]  <- 3x8: linear accel per cmd
#      [ J^-1 * T_max * (P x N)   ]  <- 3x8: angular accel per cmd

mass = lofi_dyn.vehicle.params.mass.mass_kg
inertia = np.array(lofi_dyn.vehicle.params.mass.inertia_kgm2)
num_motors = lofi_dyn.vehicle.params.motor.num_motors

# T_max is the linear thrust coefficient (poly_thrust in descending order: [c1, c0])
T_max = lofi_dyn.vehicle.params.prop.poly_thrust[0]

# Build N (thrust directions) and P (motor positions) matrices
N = np.zeros((3, num_motors))
P = np.zeros((3, num_motors))
for i in range(num_motors):
    N[:, i] = lofi_dyn.vehicle.params.motor.thrust_dir[i]
    P[:, i] = lofi_dyn.vehicle.params.motor.pos_m[i]

# Top 3 rows: linear acceleration per motor command
B0_force = (1.0 / mass) * T_max * N

# Bottom 3 rows: angular acceleration per motor command
# P x N is column-wise cross product
P_cross_N = np.zeros((3, num_motors))
for i in range(num_motors):
    P_cross_N[:, i] = np.cross(P[:, i], N[:, i])

J_inv = np.linalg.inv(inertia)
B0_moment = J_inv @ (T_max * P_cross_N)

# Full B0 matrix (6 x 8)
B0 = np.vstack([B0_force, B0_moment])
print(f"\nB0 shape: {B0.shape}")
print(f"B0:\n{B0}")

# Pseudo-inverse of B0
B0_inv = np.linalg.pinv(B0)
print(f"\nB0_inv shape: {B0_inv.shape}")

# ============================================================================
# Compute hover thrust commands
# ============================================================================
# gravity is in NED frame: [0, 0, +g] pointing down
# At hover (zero attitude), body frame = NED frame
# Gravity acceleration is [0, 0, +g] in body frame
# For hover (zero net acceleration), we need thrust acceleration to cancel gravity:
#   thrust_accel + gravity_accel = 0
#   thrust_accel = -gravity_accel = [0, 0, -g]
# The desired_accel for B0_inv is the thrust acceleration we want
g_mag = gravity[2]  # positive value (~9.8)
desired_accel = np.array(
    [0.0, 0.0, -g_mag, 0.0, 0.0, 0.0]
)  # thrust accel to cancel gravity

hover_cmds_lofi = B0_inv @ desired_accel
print(f"\nDesired accel to hover: {desired_accel}")
print(f"LoFi Hover commands: {hover_cmds_lofi}")

# Set takenoff = True so forces are applied (otherwise dynamics returns zero force)
lofi_dyn.vehicle.takenoff = True

# Verify hover: vb and omega should be ~zero after one propagate step
new_state = lofi_dyn.propagate_state(DT, lofi_dyn.vehicle.state, hover_cmds_lofi)
body_vel = new_state[v_smap_quat.body_vel].flatten()
body_omega = new_state[v_smap_quat.body_rot_rate].flatten()
print(f"LoFi Body Vel after 1 step: {body_vel}")
print(f"LoFi Body Omega after 1 step: {body_omega}")


# # ============================================================================
# # Check reaction torques at hover (VERIFIED - sigma [1,1,1,1,-1,-1,-1,-1] balances)
# # ============================================================================
# # The B0 matrix ignores reaction torques (only thrust-induced moments)
# # Now check what reaction torques we get with current sigma values
# print("\n" + "=" * 60)
# print("Checking motor reaction torques at hover")
# print("=" * 60)
#
# sigma = np.array(lofi_dyn.vehicle.params.motor.dir)
# print(f"Current sigma values: {sigma}")
#
# # Use a non-zero torque coefficient for analysis (lofi yaml has zero)
# Q_max = 0.1  # N-m per unit command (example value)
# print(f"Using Q_max = {Q_max} N-m for analysis")
#
# # Compute torque magnitude for each motor at hover command
# # Q = Q_max * |cmd|, direction determined by sigma and thrust_dir
# torque_mags = Q_max * np.abs(hover_cmds_lofi)
# print(f"Torque magnitudes: {torque_mags}")
#
# # Reaction torque for each motor: -sigma * |Q| * thrust_dir
# # (negative because reaction opposes rotor spin)
# total_reaction_torque = np.zeros(3)
# for i in range(num_motors):
#     n_i = N[:, i]
#     reaction_torque_i = -sigma[i] * torque_mags[i] * n_i
#     total_reaction_torque += reaction_torque_i
#     print(
#         f"Motor {i}: sigma={sigma[i]:+d}, Q={torque_mags[i]:.4f}, torque={reaction_torque_i}"
#     )
#
# print(f"\nTotal reaction torque: {total_reaction_torque}")
# print(f"Reaction torque magnitude: {np.linalg.norm(total_reaction_torque):.6f}")

# ============================================================================
# Compute hover commands for HiFi model and create HiFi dynamics with motor effector
# ============================================================================
# HiFi has quadratic thrust: T = T_max * u^2
# LoFi has linear thrust: T = T_max * u
# Use lofi thrust values and solve for hifi commands
print("\n" + "=" * 60)
print("Computing HiFi hover commands and creating HiFi model with motor dynamics")
print("=" * 60)

# Compute thrust from lofi commands (linear: T = T_max * u)
lofi_thrust = T_max * hover_cmds_lofi
print(f"LoFi thrust per motor: {lofi_thrust}")

# Load hifi params temporarily to get thrust polynomial
with open(hifi_config_file, "r") as f:
    hifi_params = yaml.load(f)
hifi_thrust_poly = hifi_params.prop.poly_thrust
print(f"HiFi thrust poly coeffs: {hifi_thrust_poly}")

# For quadratic T = c2*u^2, solve for u given T: u = sign(T) * sqrt(|T|/c2)
c2 = hifi_thrust_poly[0]
hover_cmds_hifi = np.sign(lofi_thrust) * np.sqrt(np.abs(lofi_thrust) / c2)
print(f"HiFi hover commands: {hover_cmds_hifi}")
print(f"Command range: [{hover_cmds_hifi.min():.4f}, {hover_cmds_hifi.max():.4f}]")

# Verify thrust produced by hifi commands
hifi_thrust_check = np.sign(hover_cmds_hifi) * c2 * hover_cmds_hifi**2
print(f"HiFi thrust check: {hifi_thrust_check}")
print(f"Thrust match: {np.allclose(lofi_thrust, hifi_thrust_check)}")

# ============================================================================
# Create HiFi dynamics with motor dynamics effector
# ============================================================================
# Motor time constant from paper: τ_mot = 0.032 seconds
TAU_MOT = 0.032
print(f"\nCreating motor dynamics effector with τ_mot = {TAU_MOT} s")

# Create motor effector with initial state at hover commands
motor_effector = MotorDynamicsEffector(
    num_motors=num_motors,
    tau_mot=TAU_MOT,
    initial_state=hover_cmds_hifi,
)
print(f"Motor effector initial state: {motor_effector.state}")

# Create HiFi dynamics with motor effector
print("Loading HiFi omnicopter with motor dynamics...")
hifi_dyn = ComplexMultirotor(str(hifi_config_file), effector=motor_effector)
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

# Verify hover: vb and omega should be ~zero after one propagate step
hifi_dyn.vehicle.takenoff = True

# With motor at hover state, commanding hover should maintain hover
print("\n" + "=" * 60)
print("Testing HiFi hover with motor dynamics")
print("=" * 60)
new_state = hifi_dyn.propagate_state(DT, hifi_dyn.vehicle.state, hover_cmds_hifi)
body_vel = new_state[v_smap_quat.body_vel].flatten()
body_omega = new_state[v_smap_quat.body_rot_rate].flatten()
print(f"Motor state after step: {motor_effector.state}")
print(f"HiFi Body Vel after 1 step: {body_vel}")
print(f"HiFi Body Omega after 1 step: {body_omega}")

# Test motor lag behavior: command zero and watch motor state decay
# print("\n" + "=" * 60)
# print("Testing motor dynamics lag behavior")
# print("=" * 60)
# print(f"Motor time constant: {TAU_MOT} s")
# print(f"DT: {DT} s")
# print(f"Expected decay per step: exp(-DT/τ) = {np.exp(-DT/TAU_MOT):.6f}")

# # Reset motor to hover state
# motor_effector.set_initial_state(hover_cmds_hifi)
# print(f"\nReset motor state to hover: {motor_effector.state[0]:.6f}")

# # Command zero for a few steps and observe decay
# cmd_zero = np.zeros(num_motors)
# for i in range(5):
#     motor_out = motor_effector.step(cmd_zero, DT)
#     print(f"Step {i+1}: motor[0] = {motor_out[0]:.6f}, expected = {hover_cmds_hifi[0] * np.exp(-DT*(i+1)/TAU_MOT):.6f}")

# # Reset for further use
# motor_effector.set_initial_state(hover_cmds_hifi)
# print(f"\nReset motor state back to hover for further tests")


# ============================================================================
# Setup INDI controller using LoFi B0 matrix
# ============================================================================
# INDI uses B0 from lofi model to handle unmodelled dynamics in hifi
print("\n" + "=" * 60)
print("Setting up INDI controller")
print("=" * 60)

# INDI controls body velocity and angular rate: x = [vb_x, vb_y, vb_z, p, q, r]
# Control law: u = u_0 + B0_inv @ (-x_dot_0 + r_dot + K @ e)
# For hover: ref = [0, 0, 0, 0, 0, 0], ref_dot = [0, 0, 0, 0, 0, 0]

# Feedback gain K (proportional gains for velocity and angular rate errors)
# Tune these for desired response
K_vel = 5.0  # Velocity error gain
K_omega = 10.0  # Angular rate error gain
K = np.diag([K_vel, K_vel, K_vel, K_omega, K_omega, K_omega])
print(f"K gains: vel={K_vel}, omega={K_omega}")

# Create INDI controller
indi_ctrl = INDI(omit_A=True)
indi_ctrl.set_state_model(dt=DT, K=K, B0=B0)
print(f"INDI B0 shape: {indi_ctrl.B0.shape}")


# ============================================================================
# Simulate hover with INDI controller
# ============================================================================
print("\n" + "=" * 60)
print("Simulating INDI hover control")
print("=" * 60)

# Add initial perturbation to make test more interesting
PERTURBED_VELOCITY = np.array([0.1, -0.05, 0.08])  # Small initial body velocity
PERTURBED_OMEGA = np.array([0.02, -0.03, 0.01])  # Small initial angular rate (rad/s)

# Reset HiFi dynamics and motor effector to initial hover state
motor_effector.set_initial_state(hover_cmds_hifi)
hifi_dyn.set_initial_conditions(
    INITIAL_POSITION,
    PERTURBED_VELOCITY,  # Start with small velocity perturbation
    INITIAL_ATTITUDE,
    PERTURBED_OMEGA,  # Start with small angular rate perturbation
    REF_LAT,
    REF_LON,
    TERRAIN_ALT,
    ned_mag,
)
hifi_dyn.vehicle.takenoff = True

print(f"Initial velocity perturbation: {PERTURBED_VELOCITY} m/s")
print(f"Initial angular rate perturbation: {np.rad2deg(PERTURBED_OMEGA)} deg/s")

# Simulation parameters
SIM_TIME_HOVER = 3.0  # seconds
num_steps = int(SIM_TIME_HOVER / DT)
print(f"Simulation time: {SIM_TIME_HOVER} s, steps: {num_steps}")

# Reference: hover in place (zero body velocity and angular rate)
ref = np.zeros(6)
ref_dot = np.zeros(6)

# Storage for plotting
time_hist = np.zeros(num_steps)
body_vel_hist = np.zeros((num_steps, 3))
body_omega_hist = np.zeros((num_steps, 3))
ned_pos_hist = np.zeros((num_steps, 3))
cmd_hist = np.zeros((num_steps, num_motors))
motor_state_hist = np.zeros((num_steps, num_motors))

# Initial conditions
cur_state = hifi_dyn.vehicle.state.copy()
cur_input = hover_cmds_hifi.copy()

# Extract initial x_dot (body accel and angular accel) from dynamics
# x_dot = [vb_dot, omega_dot] = [body_accel, body_rot_accel]
prev_body_vel = cur_state[v_smap_quat.body_vel].flatten()
prev_body_omega = cur_state[v_smap_quat.body_rot_rate].flatten()

print("Starting simulation loop...")
for ii in range(num_steps):
    tt = ii * DT
    time_hist[ii] = tt

    # Extract current state: x = [body_vel, body_omega]
    body_vel = cur_state[v_smap_quat.body_vel].flatten()
    body_omega = cur_state[v_smap_quat.body_rot_rate].flatten()
    x = np.concatenate([body_vel, body_omega])

    # Estimate x_dot from finite difference (or could use body_accel from state)
    # Using state derivatives directly is more accurate
    body_accel = cur_state[v_smap_quat.body_accel].flatten()
    body_rot_accel = cur_state[v_smap_quat.body_rot_accel].flatten()
    x_dot = np.concatenate([body_accel, body_rot_accel])

    # INDI control calculation
    u_cmd = indi_ctrl.calculate_control(
        cur_time=tt,
        cur_state=x,
        cur_state_dot=x_dot,
        cur_input=cur_input,
        ref=ref,
        ref_dot=ref_dot,
    )

    # Clamp commands to [-1, 1]
    u_cmd = np.clip(u_cmd, -1.0, 1.0)

    # Store data
    body_vel_hist[ii, :] = body_vel
    body_omega_hist[ii, :] = body_omega
    ned_pos_hist[ii, :] = cur_state[v_smap_quat.ned_pos].flatten()
    cmd_hist[ii, :] = u_cmd
    motor_state_hist[ii, :] = motor_effector.state.copy()

    # Propagate dynamics (motor effector is internal to hifi_dyn)
    cur_state = hifi_dyn.propagate_state(DT, cur_state, u_cmd).flatten()

    # Update current input for next INDI step (use actual motor state)
    cur_input = motor_effector.state.copy()

print("Simulation complete!")
print(f"Final body vel: {body_vel_hist[-1, :]}")
print(f"Final body omega: {body_omega_hist[-1, :]}")
print(f"Final NED pos: {ned_pos_hist[-1, :]}")


# ============================================================================
# Plot results and save to ValidationResults
# ============================================================================
print("\n" + "=" * 60)
print("Plotting results")
print("=" * 60)

# Create output directory
output_dir = Path(__file__).parent / "ValidationResults"
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Body velocity
ax = axes[0]
ax.plot(time_hist, body_vel_hist[:, 0], "r-", label="vb_x")
ax.plot(time_hist, body_vel_hist[:, 1], "g-", label="vb_y")
ax.plot(time_hist, body_vel_hist[:, 2], "b-", label="vb_z")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Velocity (m/s)")
ax.legend(loc="upper right")
ax.set_title("INDI Hover Control - HiFi Model with Motor Dynamics")
ax.grid(True, alpha=0.3)

# Body angular rate
ax = axes[1]
ax.plot(time_hist, np.rad2deg(body_omega_hist[:, 0]), "r-", label="p")
ax.plot(time_hist, np.rad2deg(body_omega_hist[:, 1]), "g-", label="q")
ax.plot(time_hist, np.rad2deg(body_omega_hist[:, 2]), "b-", label="r")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Body Angular Rate (deg/s)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# NED position
ax = axes[2]
ax.plot(time_hist, ned_pos_hist[:, 0], "r-", label="N")
ax.plot(time_hist, ned_pos_hist[:, 1], "g-", label="E")
ax.plot(time_hist, ned_pos_hist[:, 2], "b-", label="D")
ax.axhline(INITIAL_POSITION[2], color="b", linestyle="--", alpha=0.3, label="D_ref")
ax.set_ylabel("NED Position (m)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Motor commands
ax = axes[3]
for ii in range(num_motors):
    ax.plot(time_hist, cmd_hist[:, ii], alpha=0.7, label=f"u{ii}")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Motor Commands")
ax.legend(loc="upper right", ncol=4, fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_file = output_dir / "indi_hover_test.png"
plt.savefig(output_file, dpi=150)
print(f"Figure saved to: {output_file}")

plt.show()
