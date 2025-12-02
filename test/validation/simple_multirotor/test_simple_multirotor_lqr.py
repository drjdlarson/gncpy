"""LQR control validation for SimpleMultirotor.

This test validates the SimpleMultirotor dynamics by implementing LQR control
to command the vehicle from hover to a target waypoint.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from pathlib import Path

from gncpy.dynamics.aircraft.simple_multirotor import (
    SimpleMultirotor,
    Effector,
    v_smap,
    e_smap,
)
import gncpy.math as gmath


# ============================================================================
# Motor Dynamics Effector
# ============================================================================
class MotorDynamicsEffector(Effector):
    """Effector with first-order motor dynamics.

    Models motor spool-up/down lag: tau * omega_dot = omega_cmd - omega
    """

    def __init__(self, num_motors=4, motor_time_constant=0.05):
        """Initialize motor dynamics.

        Parameters
        ----------
        num_motors : int
            Number of motors.
        motor_time_constant : float
            Time constant in seconds (0.05s = fast response).
        """
        self.tau = motor_time_constant
        self.motor_state = np.zeros(num_motors)

    def step(self, input_cmds, dt=0.01):
        """Apply first-order lag to motor commands.

        Parameters
        ----------
        input_cmds : numpy array
            Desired motor commands.
        dt : float
            Timestep in seconds.

        Returns
        -------
        numpy array
            Actual motor commands after lag.
        """
        # First-order response: state += (cmd - state) * dt / tau
        alpha = dt / self.tau
        self.motor_state += alpha * (input_cmds - self.motor_state)
        return self.motor_state.copy()


# ============================================================================
# Configuration
# ============================================================================
DT = 0.01
SIM_TIME = 10.0

# Initial conditions (hover)
INIT_POS = np.array([0.0, 0.0, -2.0])  # NED (m)
INIT_VEL = np.array([0.0, 0.0, 0.0])  # body frame (m/s)
INIT_EULER = np.array([0.0, 0.0, 0.0])  # yaw, pitch, roll (deg)
INIT_RATES = np.array([0.0, 0.0, 0.0])  # body rates (rad/s)

REF_LAT, REF_LON, TERRAIN_ALT = 34.0, -86.0, 0.0
TARGET_POS = np.array([2.0, 1.0, -2.5])  # NED (m)

# LQR weights
Q = scipy.linalg.block_diag(
    np.diag([1.0, 1.0, 1.0]),  # position
    np.diag([0.1, 0.1, 0.1]),  # velocity
    np.diag([0.1, 0.1, 0.1]),  # attitude
    np.diag([0.1, 0.1, 0.1]),  # rates
)
R = np.diag([1.0, 1.0, 1.0, 1.0])  # motor commands


# ============================================================================
# Initialize Vehicle with Motor Dynamics
# ============================================================================
print("Initializing vehicle with motor dynamics...")
config_file = Path(__file__).parent / "small_quad_config.yaml"

# Create motor dynamics effector (50ms time constant = fast but realistic)
motor_effector = MotorDynamicsEffector(num_motors=4, motor_time_constant=0.05)
vehicle = SimpleMultirotor(str(config_file), effector=motor_effector)

ned_mag = np.array([20.0, 5.0, 45.0])
vehicle.set_initial_conditions(
    INIT_POS, INIT_VEL, INIT_EULER, INIT_RATES, REF_LAT, REF_LON, TERRAIN_ALT, ned_mag
)
vehicle.vehicle.takenoff = True

# Calculate hover motor command
mass = vehicle.vehicle.params.mass.mass_kg
gravity = 9.81
hover_thrust_per_motor = mass * gravity / vehicle.vehicle.params.motor.num_motors

thrust_poly = np.polynomial.Polynomial(vehicle.vehicle.params.prop.poly_thrust[-1::-1])
roots = (thrust_poly - hover_thrust_per_motor).roots()
valid_roots = roots[np.isreal(roots) & (roots.real > 0) & (roots.real < 1)]
hover_cmd = float(valid_roots[0].real) if len(valid_roots) > 0 else 0.7

trim_motor_cmd = np.ones(vehicle.vehicle.params.motor.num_motors) * hover_cmd
print(f"Hover command: {hover_cmd:.3f}, Mass: {mass:.3f} kg")


# ============================================================================
# Linearization
# ============================================================================
print("\nLinearizing about hover...")


def state_deriv(x, u, t):
    """12-DOF dynamics: [ned_pos(3), body_vel(3), euler(3), body_rates(3)]."""
    old_state = vehicle.vehicle.state.copy()

    # Set state
    vehicle.vehicle.state[v_smap.ned_pos] = x[0:3]
    vehicle.vehicle.state[v_smap.body_vel] = x[3:6]
    eul_inds = v_smap.roll + v_smap.pitch + v_smap.yaw
    vehicle.vehicle.state[eul_inds] = x[6:9]
    vehicle.vehicle.state[v_smap.body_rot_rate] = x[9:12]

    # Get forces
    dcm_e2b = vehicle.vehicle.eul_to_dcm(x[8], x[7], x[6])
    gravity_body = dcm_e2b @ np.array([0, 0, gravity * mass])
    prop_force, prop_mom = vehicle.vehicle._calc_prop_force_mom(u)
    force = gravity_body + prop_force

    # Compute derivatives
    dcm = vehicle.vehicle.eul_to_dcm(x[8], x[7], x[6])
    ned_vel = dcm.T @ x[3:6]
    body_accel = force / mass + np.cross(x[3:6], x[9:12])

    # Euler rates
    s_phi, c_phi = np.sin(x[6]), np.cos(x[6])
    t_theta, c_theta = np.tan(x[7]), np.cos(x[7])
    c_theta = max(abs(c_theta), 1e-6) * np.sign(c_theta) if c_theta != 0 else 1e-6
    eul_dot_mat = np.array(
        [
            [1, s_phi * t_theta, c_phi * t_theta],
            [0, c_phi, -s_phi],
            [0, s_phi / c_theta, c_phi / c_theta],
        ]
    )
    euler_rates = eul_dot_mat @ x[9:12]

    inertia = np.array(vehicle.vehicle.params.mass.inertia_kgm2)
    body_rot_accel = np.linalg.inv(inertia) @ (
        prop_mom - np.cross(x[9:12], inertia @ x[9:12])
    )

    vehicle.vehicle.state = old_state
    return np.concatenate([ned_vel, body_accel, euler_rates, body_rot_accel])


# Get trim state
eul_inds = v_smap.roll + v_smap.pitch + v_smap.yaw
trim_state = np.concatenate(
    [
        vehicle.vehicle.state[v_smap.ned_pos],
        vehicle.vehicle.state[v_smap.body_vel],
        vehicle.vehicle.state[eul_inds],
        vehicle.vehicle.state[v_smap.body_rot_rate],
    ]
)

# Linearize using gncpy.math
A, B = gmath.linearize_dynamics(state_deriv, trim_state, trim_motor_cmd)

print(f"A shape: {A.shape}, B shape: {B.shape}")
print(
    f"System rank: {np.linalg.matrix_rank(np.hstack([B] + [np.linalg.matrix_power(A, i) @ B for i in range(1, 12)]))}/12"
)


# ============================================================================
# Design LQR
# ============================================================================
print("Designing LQR...")
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print(f"K shape: {K.shape}, Max gain: {np.max(np.abs(K)):.4f}")


# ============================================================================
# Simulate
# ============================================================================
print("\nRunning simulation...")

vehicle.set_initial_conditions(
    INIT_POS, INIT_VEL, INIT_EULER, INIT_RATES, REF_LAT, REF_LON, TERRAIN_ALT, ned_mag
)
vehicle.vehicle.takenoff = True

time_hist, pos_hist, vel_hist, att_hist, rate_hist, motor_hist = [], [], [], [], [], []
x_target = np.array(
    [TARGET_POS[0], TARGET_POS[1], TARGET_POS[2], 0, 0, 0, 0, 0, 0, 0, 0, 0]
)

time = 0.0
for i in range(int(SIM_TIME / DT)):
    x_current = np.concatenate(
        [
            vehicle.vehicle.state[v_smap.ned_pos],
            vehicle.vehicle.state[v_smap.body_vel],
            vehicle.vehicle.state[eul_inds],
            vehicle.vehicle.state[v_smap.body_rot_rate],
        ]
    )

    # Hover for first 0.1s, then engage LQR
    if i < 10:
        motor_cmd = trim_motor_cmd.copy()
    else:
        delta_u = -K @ (x_current - x_target)
        motor_cmd = np.clip(trim_motor_cmd + delta_u, 0.1, 0.95)

    vehicle.propagate_state(DT, vehicle.vehicle.state, u=motor_cmd)

    time_hist.append(time)
    pos_hist.append(vehicle.vehicle.state[v_smap.ned_pos].copy())
    vel_hist.append(vehicle.vehicle.state[v_smap.body_vel].copy())
    att_hist.append(vehicle.vehicle.state[eul_inds].copy())
    rate_hist.append(vehicle.vehicle.state[v_smap.body_rot_rate].copy())
    motor_hist.append(motor_cmd.copy())

    time += DT

# Convert to arrays
time_hist = np.array(time_hist)
pos_hist = np.array(pos_hist)
vel_hist = np.array(vel_hist)
att_hist = np.array(att_hist)
rate_hist = np.array(rate_hist)
motor_hist = np.array(motor_hist)

print(f"\nSimulation complete!")
print(f"Final position: {pos_hist[-1]}")
print(f"Target position: {TARGET_POS}")
print(f"Position error: {np.linalg.norm(pos_hist[-1] - TARGET_POS):.4f} m")


# ============================================================================
# Plot
# ============================================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Position
axes[0, 0].plot(time_hist, pos_hist[:, 0], label="North")
axes[0, 0].plot(time_hist, pos_hist[:, 1], label="East")
axes[0, 0].plot(time_hist, pos_hist[:, 2], label="Down")
axes[0, 0].axhline(TARGET_POS[0], color="r", linestyle="--", alpha=0.5)
axes[0, 0].axhline(TARGET_POS[1], color="g", linestyle="--", alpha=0.5)
axes[0, 0].axhline(TARGET_POS[2], color="b", linestyle="--", alpha=0.5)
axes[0, 0].set_ylabel("Position (m)")
axes[0, 0].legend()
axes[0, 0].grid(True)
axes[0, 0].set_title("NED Position")

# Velocity
axes[0, 1].plot(time_hist, vel_hist)
axes[0, 1].set_ylabel("Velocity (m/s)")
axes[0, 1].legend(["u", "v", "w"])
axes[0, 1].grid(True)
axes[0, 1].set_title("Body Velocity")

# Attitude
axes[1, 0].plot(time_hist, np.rad2deg(att_hist))
axes[1, 0].set_ylabel("Attitude (deg)")
axes[1, 0].legend(["Roll", "Pitch", "Yaw"])
axes[1, 0].grid(True)
axes[1, 0].set_title("Euler Angles")

# Rates
axes[1, 1].plot(time_hist, np.rad2deg(rate_hist))
axes[1, 1].set_ylabel("Rate (deg/s)")
axes[1, 1].legend(["p", "q", "r"])
axes[1, 1].grid(True)
axes[1, 1].set_title("Body Rates")

# Motor commands
axes[2, 0].plot(time_hist, motor_hist)
axes[2, 0].axhline(hover_cmd, color="k", linestyle="--", alpha=0.5)
axes[2, 0].set_ylabel("Motor Command")
axes[2, 0].set_xlabel("Time (s)")
axes[2, 0].set_ylim([0, 1])
axes[2, 0].legend(["M1", "M2", "M3", "M4", "Hover"])
axes[2, 0].grid(True)
axes[2, 0].set_title("Motor Commands")

# 3D trajectory
ax3d = fig.add_subplot(3, 2, 6, projection="3d")
ax3d.plot(pos_hist[:, 1], pos_hist[:, 0], -pos_hist[:, 2], "b-")
ax3d.scatter([0], [0], [2], color="g", s=100, marker="o", label="Start")
ax3d.scatter(
    [TARGET_POS[1]],
    [TARGET_POS[0]],
    [-TARGET_POS[2]],
    color="r",
    s=100,
    marker="x",
    label="Target",
)
ax3d.set_xlabel("East (m)")
ax3d.set_ylabel("North (m)")
ax3d.set_zlabel("Up (m)")
ax3d.legend()
ax3d.set_title("3D Trajectory")

plt.tight_layout()
output_file = Path(__file__).parent / "lqr_results.png"
plt.savefig(output_file, dpi=150)
print(f"Plot saved: {output_file}")

print("\nValidation complete!")
