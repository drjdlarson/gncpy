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
)
import gncpy.math as gmath

from gncpy.control.INDI import INDI


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


# Load Low Fidelity Omnicopter Dynamics
print("Loading low fidelity omnicopter...")
lofi_config_file = Path(__file__).parent / "omnicopter_config.yaml"
hifi_config_file = Path(__file__).parent / "omnicopter_config_hifi.yaml"

lofi_dyn = ComplexMultirotor(str(lofi_config_file))
hifi_dyn = ComplexMultirotor(str(hifi_config_file))

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
# Compute hover commands for HiFi model
# ============================================================================
# HiFi has quadratic thrust: T = T_max * u^2
# LoFi has linear thrust: T = T_max * u
# Use lofi thrust values and solve for hifi commands
print("\n" + "=" * 60)
print("Computing HiFi hover commands")
print("=" * 60)

# Compute thrust from lofi commands (linear: T = T_max * u)
lofi_thrust = T_max * hover_cmds_lofi
print(f"LoFi thrust per motor: {lofi_thrust}")

# HiFi thrust polynomial (descending order: [c2, c1, c0] for T = c2*u^2 + c1*u + c0)
hifi_thrust_poly = hifi_dyn.vehicle.params.prop.poly_thrust
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

# Verify hover: vb and omega should be ~zero after one propagate step
hifi_dyn.vehicle.takenoff = True
new_state = hifi_dyn.propagate_state(DT, hifi_dyn.vehicle.state, hover_cmds_hifi)
body_vel = new_state[v_smap_quat.body_vel].flatten()
body_omega = new_state[v_smap_quat.body_rot_rate].flatten()
print(f"HiFi Body Vel after 1 step: {body_vel}")
print(f"HiFi Body Omega after 1 step: {body_omega}")
