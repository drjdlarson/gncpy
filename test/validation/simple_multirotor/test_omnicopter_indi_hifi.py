"""INDI rate control validation for omnicopter with HIGH-FIDELITY model.

This test validates INDI robustness when the actual plant differs from the
linear design model (B0). The high-fidelity model has:
- Quadratic thrust: T = T_max * u^2
- Aerodynamic drag (cd = 0.3)
- Reaction torques from rotor spin
- Motor dynamics (first-order lag, tau = 0.03s)

Two test scenarios:
1. Circular flight while rolling - commands NED velocity in a circle while
   simultaneously commanding body roll rate
2. Flight path with rate commands - various velocity and rate maneuvers

INDI tracks body velocity and angular rate references.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from gncpy.dynamics.aircraft.complex_multirotor import (
    ComplexMultirotor,
    v_smap_quat,
)
from gncpy.dynamics.aircraft.simple_multirotor import Effector
from gncpy.control.INDI import INDI
import gncpy.math as gmath


# ============================================================================
# Motor Dynamics Effector
# ============================================================================
class MotorDynamicsEffector(Effector):
    """Effector with first-order motor dynamics."""

    def __init__(self, num_motors=8, motor_time_constant=0.03):
        self.tau = motor_time_constant
        self.motor_state = np.zeros(num_motors)

    def step(self, input_cmds, dt=0.01):
        alpha = dt / self.tau
        self.motor_state += alpha * (input_cmds - self.motor_state)
        return self.motor_state.copy()

    def reset(self, initial_state):
        self.motor_state = initial_state.copy()


# ============================================================================
# Omnicopter Parameters (for B0 computation - LINEAR design model)
# ============================================================================
MASS = 0.886  # kg
T_MAX = 6.7  # N (linear approximation of thrust effectiveness)
J = np.diag([0.005, 0.005, 0.005])  # kg*m^2
J_INV = np.linalg.inv(J)

# Position matrix P: motor positions
SCALE = 0.184 / np.sqrt(3)
P = SCALE * np.array(
    [
        [1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
    ]
)

# Thrust direction matrix N (matches paper exactly)
a = 0.5 + 1 / np.sqrt(12)
b = 0.5 - 1 / np.sqrt(12)
c = 1 / np.sqrt(3)

N = np.array(
    [
        [-a, b, -b, a, a, -b, b, -a],
        [b, a, -a, -b, -b, -a, a, b],
        [c, -c, -c, c, c, -c, -c, c],  # Corrected Z components
    ]
)

for i in range(N.shape[1]):
    N[:, i] = N[:, i] / np.linalg.norm(N[:, i])


def compute_B0():
    """Compute the LINEAR control effectiveness matrix B0."""
    B_force = (T_MAX / MASS) * N
    P_cross_N = np.zeros((3, 8))
    for i in range(8):
        P_cross_N[:, i] = np.cross(P[:, i], N[:, i])
    B_torque = J_INV @ (T_MAX * P_cross_N)
    return np.vstack([B_force, B_torque])


B0 = compute_B0()
B0_pinv = B0.T @ np.linalg.inv(B0 @ B0.T)


# ============================================================================
# Test 0: Pure Angular Rate Tracking (No Velocity Commands)
# ============================================================================
def test_pure_rate_tracking():
    """Test ONLY angular rate tracking with zero velocity command.

    This isolates rate tracking performance from velocity/attitude coupling.
    """

    print("=" * 70)
    print("TEST 0: PURE ANGULAR RATE TRACKING")
    print("=" * 70)
    print("\nCommanding ONLY roll rate, zero velocity.")
    print("This isolates angular rate tracking from velocity coupling.")

    # Configuration
    DT = 0.01
    SIM_TIME = 10.0

    # Roll rate - reduced from 36 to 18 deg/s
    roll_rate = np.deg2rad(18)  # 18 deg/s = one full rotation in 20s

    # Low-pass filter cutoff for omega_dot (exponential smoothing)
    # alpha = DT / (tau + DT), where tau = 1/(2*pi*fc)
    fc_omega_dot = 5.0  # 5 Hz cutoff
    tau_filter = 1.0 / (2.0 * np.pi * fc_omega_dot)
    alpha_filter = DT / (tau_filter + DT)

    # INDI gains - MUST BE LOW to avoid saturation!
    # B0 has large entries (~140-194 rad/s² per unit input)
    # K_omega = 1.0 means: 1 rad/s error -> 1 rad/s² desired acceleration
    K_vel = 0.5  # Reduced from 2.0
    K_omega = 1.0  # Reduced from 5.0 - avoids saturation
    K = np.diag([K_vel, K_vel, K_vel, K_omega, K_omega, K_omega])

    # Initialize vehicle with HIGH-FIDELITY model
    config_file = Path(__file__).parent / "omnicopter_config_hifi.yaml"
    motor_effector = MotorDynamicsEffector(
        num_motors=8, motor_time_constant=0.02
    )  # Faster motors
    vehicle = ComplexMultirotor(str(config_file), effector=motor_effector)

    init_pos = np.array([0.0, 0.0, -3.0])
    init_vel = np.array([0.0, 0.0, 0.0])
    init_euler = np.array([0.0, 0.0, 0.0])
    init_rates = np.array([0.0, 0.0, 0.0])

    vehicle.set_initial_conditions(
        init_pos,
        init_vel,
        init_euler,
        init_rates,
        34.0,
        -86.0,
        0.0,
        np.array([20.0, 5.0, 45.0]),
    )
    vehicle.vehicle.takenoff = True

    # Initialize INDI
    indi = INDI(omit_A=True)
    indi.set_state_model(K=K, H=None, B0=B0)

    # Initial motor state - START FROM HOVER!
    # Hover scale s = 0.53 exactly balances gravity (thrust = 8.69 N = weight)
    # With quadratic thrust T = 6.7*u*|u|, INDI needs to operate around hover
    u_hover = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * 0.53
    u = u_hover.copy()
    motor_effector.reset(u)

    # History
    time_hist = []
    omega_hist = []
    ref_omega_hist = []
    omega_dot_hist = []
    omega_dot_raw_hist = []
    motor_hist = []
    euler_hist = []
    vel_body_hist = []

    prev_vel = init_vel.copy()
    prev_omega = init_rates.copy()
    omega_dot_filtered = np.zeros(3)  # Initialize filtered derivative
    vel_dot_filtered = np.zeros(3)

    time = 0.0

    print(
        f"\nSimulating {SIM_TIME}s with roll rate = {np.rad2deg(roll_rate):.1f} deg/s..."
    )
    print(
        f"Using low-pass filter on omega_dot: fc = {fc_omega_dot:.1f} Hz, alpha = {alpha_filter:.3f}"
    )
    print("\nDebug: omega_dot samples (should be small, ~0 at steady state):")

    for i in range(int(SIM_TIME / DT)):
        # Current state
        pos = vehicle.vehicle.state[v_smap_quat.ned_pos].copy()
        vel_B = vehicle.vehicle.state[v_smap_quat.body_vel].copy()
        quat = vehicle.vehicle.state[v_smap_quat.quat].copy()
        omega_B = vehicle.vehicle.state[v_smap_quat.body_rot_rate].copy()

        roll, pitch, yaw = gmath.quat_to_euler(quat)

        # Reference: ZERO velocity, constant roll rate only
        v_body_ref = np.array([0.0, 0.0, 0.0])
        omega_ref = np.array([roll_rate, 0.0, 0.0])

        ref = np.concatenate([v_body_ref, omega_ref])
        ref_dot = np.zeros(6)  # Constant reference, no acceleration feedforward

        # Current state for INDI
        x = np.concatenate([vel_B, omega_B])

        # State derivatives (finite difference)
        vel_dot_raw = (vel_B - prev_vel) / DT
        omega_dot_raw = (omega_B - prev_omega) / DT

        # Low-pass filter the derivatives to reduce noise
        vel_dot_filtered = (
            alpha_filter * vel_dot_raw + (1 - alpha_filter) * vel_dot_filtered
        )
        omega_dot_filtered = (
            alpha_filter * omega_dot_raw + (1 - alpha_filter) * omega_dot_filtered
        )

        x_dot = np.concatenate([vel_dot_filtered, omega_dot_filtered])

        # Debug: print omega_dot every second
        if i % 100 == 0 and i > 0:
            print(
                f"  t={time:5.1f}s: omega_dot_raw = [{np.rad2deg(omega_dot_raw[0]):7.1f}, {np.rad2deg(omega_dot_raw[1]):7.1f}, {np.rad2deg(omega_dot_raw[2]):7.1f}], filtered = [{np.rad2deg(omega_dot_filtered[0]):7.1f}, {np.rad2deg(omega_dot_filtered[1]):7.1f}, {np.rad2deg(omega_dot_filtered[2]):7.1f}] deg/s²"
            )

        # Record history
        time_hist.append(time)
        omega_hist.append(omega_B.copy())
        ref_omega_hist.append(omega_ref.copy())
        omega_dot_hist.append(omega_dot_filtered.copy())
        omega_dot_raw_hist.append(omega_dot_raw.copy())
        motor_hist.append(u.copy())
        euler_hist.append(np.array([roll, pitch, yaw]))
        vel_body_hist.append(vel_B.copy())

        # INDI control
        if i > 0:
            u = indi.calculate_control(
                cur_time=time,
                cur_state=x,
                cur_state_dot=x_dot,
                cur_input=u,
                ref=ref,
                ref_dot=ref_dot,
            )
            u = np.clip(u, -1.0, 1.0)

        prev_vel = vel_B.copy()
        prev_omega = omega_B.copy()

        vehicle.propagate_state(DT, vehicle.vehicle.state, u=u)
        time += DT

    # Convert to arrays
    time_hist = np.array(time_hist)
    omega_hist = np.array(omega_hist)
    ref_omega_hist = np.array(ref_omega_hist)
    omega_dot_hist = np.array(omega_dot_hist)
    omega_dot_raw_hist = np.array(omega_dot_raw_hist)
    motor_hist = np.array(motor_hist)
    euler_hist = np.array(euler_hist)
    vel_body_hist = np.array(vel_body_hist)

    # Compute errors
    omega_error = omega_hist - ref_omega_hist
    rms_omega_error = np.sqrt(np.mean(omega_error**2, axis=0))

    # omega_dot statistics (to check noise)
    omega_dot_std = np.std(omega_dot_hist, axis=0)
    omega_dot_max = np.max(np.abs(omega_dot_hist), axis=0)
    omega_dot_raw_std = np.std(omega_dot_raw_hist, axis=0)
    omega_dot_raw_max = np.max(np.abs(omega_dot_raw_hist), axis=0)

    print(f"\n--- Results ---")
    print(
        f"RMS angular rate error: p={np.rad2deg(rms_omega_error[0]):.2f}, q={np.rad2deg(rms_omega_error[1]):.2f}, r={np.rad2deg(rms_omega_error[2]):.2f} deg/s"
    )
    print(
        f"omega_dot RAW std: p={np.rad2deg(omega_dot_raw_std[0]):.1f}, q={np.rad2deg(omega_dot_raw_std[1]):.1f}, r={np.rad2deg(omega_dot_raw_std[2]):.1f} deg/s²"
    )
    print(
        f"omega_dot FILTERED std: p={np.rad2deg(omega_dot_std[0]):.1f}, q={np.rad2deg(omega_dot_std[1]):.1f}, r={np.rad2deg(omega_dot_std[2]):.1f} deg/s²"
    )
    print(
        f"omega_dot RAW max: p={np.rad2deg(omega_dot_raw_max[0]):.1f}, q={np.rad2deg(omega_dot_raw_max[1]):.1f}, r={np.rad2deg(omega_dot_raw_max[2]):.1f} deg/s²"
    )
    print(
        f"omega_dot FILTERED max: p={np.rad2deg(omega_dot_max[0]):.1f}, q={np.rad2deg(omega_dot_max[1]):.1f}, r={np.rad2deg(omega_dot_max[2]):.1f} deg/s²"
    )
    print(
        f"Final roll: {np.rad2deg(euler_hist[-1, 0]):.1f}° (expected: {np.rad2deg(roll_rate * SIM_TIME):.1f}°)"
    )
    print(f"Final pitch: {np.rad2deg(euler_hist[-1, 1]):.2f}° (should be ~0)")
    print(f"Final yaw: {np.rad2deg(euler_hist[-1, 2]):.2f}° (should be ~0)")

    # Save figures
    results_dir = Path(__file__).parent / "ValidationResults"
    results_dir.mkdir(exist_ok=True)

    # Figure: Angular Rate Tracking
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(
        time_hist, np.rad2deg(omega_hist[:, 0]), "b-", label="Actual", linewidth=2
    )
    axes[0].plot(
        time_hist,
        np.rad2deg(ref_omega_hist[:, 0]),
        "r--",
        label="Reference",
        linewidth=2,
        alpha=0.8,
    )
    axes[0].set_ylabel("p (deg/s)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Pure Rate Tracking: Angular Rate", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, np.rad2deg(omega_hist[:, 1]), "b-", linewidth=2)
    axes[1].plot(
        time_hist, np.rad2deg(ref_omega_hist[:, 1]), "r--", linewidth=2, alpha=0.8
    )
    axes[1].set_ylabel("q (deg/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, np.rad2deg(omega_hist[:, 2]), "b-", linewidth=2)
    axes[2].plot(
        time_hist, np.rad2deg(ref_omega_hist[:, 2]), "r--", linewidth=2, alpha=0.8
    )
    axes[2].set_ylabel("r (deg/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "pure_rate_tracking.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: omega_dot (to check noise)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_hist, np.rad2deg(omega_dot_hist[:, 0]), "b-", linewidth=1)
    axes[0].set_ylabel("p_dot (deg/s²)")
    axes[0].set_title(
        "Pure Rate Tracking: Angular Acceleration (omega_dot)", fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, np.rad2deg(omega_dot_hist[:, 1]), "g-", linewidth=1)
    axes[1].set_ylabel("q_dot (deg/s²)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, np.rad2deg(omega_dot_hist[:, 2]), "r-", linewidth=1)
    axes[2].set_ylabel("r_dot (deg/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "pure_rate_omega_dot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: Euler Angles
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_hist, np.rad2deg(euler_hist[:, 0]), "b-", linewidth=2)
    axes[0].set_ylabel("Roll (deg)")
    axes[0].set_title("Pure Rate Tracking: Euler Angles", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, np.rad2deg(euler_hist[:, 1]), "g-", linewidth=2)
    axes[1].set_ylabel("Pitch (deg)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, np.rad2deg(euler_hist[:, 2]), "r-", linewidth=2)
    axes[2].set_ylabel("Yaw (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "pure_rate_euler.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: Motor Commands
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for m in range(8):
        ax.plot(
            time_hist, motor_hist[:, m], color=colors[m], label=f"M{m+1}", linewidth=1.5
        )
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(-1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("Motor Command")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc="upper right", ncol=4)
    ax.set_title("Pure Rate Tracking: Motor Commands", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "pure_rate_motors.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nFigures saved to {results_dir}/")

    return {
        "rms_omega_error": rms_omega_error,
        "omega_dot_std": omega_dot_std,
        "omega_dot_max": omega_dot_max,
    }


# ============================================================================
# Test 1: Circular Flight While Rolling
# ============================================================================
def test_circular_flight_with_roll():
    """Fly in a circle (NED velocity) while continuously rolling."""

    print("=" * 70)
    print("TEST 1: CIRCULAR FLIGHT WHILE ROLLING")
    print("=" * 70)
    print("\nCommanding circular velocity in NED frame while rolling.")
    print("This tests INDI's ability to decouple translation and rotation.")

    # Configuration
    DT = 0.01
    SIM_TIME = 20.0

    # Circle parameters
    circle_radius = 1.0  # m/s velocity magnitude
    circle_period = 10.0  # seconds per revolution
    circle_freq = 1.0 / circle_period

    # Roll rate
    roll_rate = np.deg2rad(36)  # 36 deg/s = one full rotation in 10s

    # INDI gains - MUST BE LOW to avoid saturation!
    K_vel = 0.5  # Reduced from 2.0
    K_omega = 1.0  # Reduced from 5.0
    K = np.diag([K_vel, K_vel, K_vel, K_omega, K_omega, K_omega])

    # Initialize vehicle with HIGH-FIDELITY model
    config_file = Path(__file__).parent / "omnicopter_config_hifi.yaml"
    motor_effector = MotorDynamicsEffector(num_motors=8, motor_time_constant=0.03)
    vehicle = ComplexMultirotor(str(config_file), effector=motor_effector)

    init_pos = np.array([0.0, 0.0, -3.0])
    init_vel = np.array([0.0, 0.0, 0.0])
    init_euler = np.array([0.0, 0.0, 0.0])
    init_rates = np.array([0.0, 0.0, 0.0])

    vehicle.set_initial_conditions(
        init_pos,
        init_vel,
        init_euler,
        init_rates,
        34.0,
        -86.0,
        0.0,
        np.array([20.0, 5.0, 45.0]),
    )
    vehicle.vehicle.takenoff = True

    # Initialize INDI
    indi = INDI(omit_A=True)
    indi.set_state_model(K=K, H=None, B0=B0)

    # Initial motor state - START FROM HOVER!
    # Hover scale s = 0.53 exactly balances gravity
    u_hover = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * 0.53
    u = u_hover.copy()
    motor_effector.reset(u)

    # History
    time_hist = []
    pos_hist = []
    vel_body_hist = []
    vel_ned_hist = []
    omega_hist = []
    ref_vel_ned_hist = []
    ref_omega_hist = []
    motor_hist = []
    euler_hist = []

    prev_vel = init_vel.copy()
    prev_omega = init_rates.copy()

    time = 0.0

    print(f"\nSimulating {SIM_TIME}s...")

    for i in range(int(SIM_TIME / DT)):
        # Current state
        pos = vehicle.vehicle.state[v_smap_quat.ned_pos].copy()
        vel_B = vehicle.vehicle.state[v_smap_quat.body_vel].copy()
        quat = vehicle.vehicle.state[v_smap_quat.quat].copy()
        omega_B = vehicle.vehicle.state[v_smap_quat.body_rot_rate].copy()

        # Get NED velocity for logging
        vel_ned = gmath.quat_rotate_vector(quat, vel_B)

        # Euler angles for logging
        roll, pitch, yaw = gmath.quat_to_euler(quat)

        # Reference: circular velocity in NED + roll rate in body
        v_ned_ref = np.array(
            [
                circle_radius * np.cos(2 * np.pi * circle_freq * time),
                circle_radius * np.sin(2 * np.pi * circle_freq * time),
                0.0,  # Hold altitude
            ]
        )

        omega_ref = np.array([roll_rate, 0.0, 0.0])  # Roll only

        # Convert NED velocity to body frame for INDI
        q_inv = gmath.quat_conjugate(quat)
        v_body_ref = gmath.quat_rotate_vector(q_inv, v_ned_ref)

        # INDI reference
        ref = np.concatenate([v_body_ref, omega_ref])

        # Reference derivatives (for feedforward)
        v_ned_ref_dot = np.array(
            [
                -circle_radius
                * 2
                * np.pi
                * circle_freq
                * np.sin(2 * np.pi * circle_freq * time),
                circle_radius
                * 2
                * np.pi
                * circle_freq
                * np.cos(2 * np.pi * circle_freq * time),
                0.0,
            ]
        )
        # Transform to body (approximate - ignoring rotation rate coupling for simplicity)
        v_body_ref_dot = gmath.quat_rotate_vector(q_inv, v_ned_ref_dot)
        omega_ref_dot = np.array([0.0, 0.0, 0.0])
        ref_dot = np.concatenate([v_body_ref_dot, omega_ref_dot])

        # Current state for INDI
        x = np.concatenate([vel_B, omega_B])

        # State derivatives
        vel_dot = (vel_B - prev_vel) / DT
        omega_dot = (omega_B - prev_omega) / DT
        x_dot = np.concatenate([vel_dot, omega_dot])

        # Record history
        time_hist.append(time)
        pos_hist.append(pos.copy())
        vel_body_hist.append(vel_B.copy())
        vel_ned_hist.append(vel_ned.copy())
        omega_hist.append(omega_B.copy())
        ref_vel_ned_hist.append(v_ned_ref.copy())
        ref_omega_hist.append(omega_ref.copy())
        motor_hist.append(u.copy())
        euler_hist.append(np.array([roll, pitch, yaw]))

        # INDI control
        if i > 0:
            u = indi.calculate_control(
                cur_time=time,
                cur_state=x,
                cur_state_dot=x_dot,
                cur_input=u,
                ref=ref,
                ref_dot=ref_dot,
            )
            u = np.clip(u, -1.0, 1.0)

        prev_vel = vel_B.copy()
        prev_omega = omega_B.copy()

        vehicle.propagate_state(DT, vehicle.vehicle.state, u=u)
        time += DT

    # Convert to arrays
    time_hist = np.array(time_hist)
    pos_hist = np.array(pos_hist)
    vel_body_hist = np.array(vel_body_hist)
    vel_ned_hist = np.array(vel_ned_hist)
    omega_hist = np.array(omega_hist)
    ref_vel_ned_hist = np.array(ref_vel_ned_hist)
    ref_omega_hist = np.array(ref_omega_hist)
    motor_hist = np.array(motor_hist)
    euler_hist = np.array(euler_hist)

    # Compute errors
    vel_ned_error = vel_ned_hist - ref_vel_ned_hist
    omega_error = omega_hist - ref_omega_hist

    rms_vel_error = np.sqrt(np.mean(vel_ned_error**2, axis=0))
    rms_omega_error = np.sqrt(np.mean(omega_error**2, axis=0))

    print(f"\nResults:")
    print(
        f"  RMS NED velocity error: N={rms_vel_error[0]:.3f}, E={rms_vel_error[1]:.3f}, D={rms_vel_error[2]:.3f} m/s"
    )
    print(
        f"  RMS angular rate error: p={np.rad2deg(rms_omega_error[0]):.2f}, q={np.rad2deg(rms_omega_error[1]):.2f}, r={np.rad2deg(rms_omega_error[2]):.2f} deg/s"
    )
    print(
        f"  Final roll: {np.rad2deg(euler_hist[-1, 0]):.1f}° (expected: {np.rad2deg(roll_rate * SIM_TIME) % 360:.1f}°)"
    )

    # Save figures
    results_dir = Path(__file__).parent / "ValidationResults"
    results_dir.mkdir(exist_ok=True)

    # Figure 1: NED Velocity Tracking
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_hist, vel_ned_hist[:, 0], "b-", label="Actual", linewidth=2)
    axes[0].plot(
        time_hist,
        ref_vel_ned_hist[:, 0],
        "r--",
        label="Reference",
        linewidth=2,
        alpha=0.8,
    )
    axes[0].set_ylabel("North (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        "Circular Flight: NED Velocity Tracking (while rolling)", fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, vel_ned_hist[:, 1], "b-", linewidth=2)
    axes[1].plot(time_hist, ref_vel_ned_hist[:, 1], "r--", linewidth=2, alpha=0.8)
    axes[1].set_ylabel("East (m/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, vel_ned_hist[:, 2], "b-", linewidth=2)
    axes[2].plot(time_hist, ref_vel_ned_hist[:, 2], "r--", linewidth=2, alpha=0.8)
    axes[2].set_ylabel("Down (m/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        results_dir / "circular_velocity_tracking.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(results_dir / "circular_velocity_tracking.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: Angular Rate Tracking
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(
        time_hist, np.rad2deg(omega_hist[:, 0]), "b-", label="Actual", linewidth=2
    )
    axes[0].plot(
        time_hist,
        np.rad2deg(ref_omega_hist[:, 0]),
        "r--",
        label="Reference",
        linewidth=2,
        alpha=0.8,
    )
    axes[0].set_ylabel("p (deg/s)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Circular Flight: Angular Rate Tracking", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, np.rad2deg(omega_hist[:, 1]), "b-", linewidth=2)
    axes[1].plot(
        time_hist, np.rad2deg(ref_omega_hist[:, 1]), "r--", linewidth=2, alpha=0.8
    )
    axes[1].set_ylabel("q (deg/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, np.rad2deg(omega_hist[:, 2]), "b-", linewidth=2)
    axes[2].plot(
        time_hist, np.rad2deg(ref_omega_hist[:, 2]), "r--", linewidth=2, alpha=0.8
    )
    axes[2].set_ylabel("r (deg/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        results_dir / "circular_rate_tracking.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(results_dir / "circular_rate_tracking.pdf", bbox_inches="tight")
    plt.close()

    # Figure 3: Euler Angles
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_hist, np.rad2deg(euler_hist[:, 0]), "b-", linewidth=2)
    axes[0].set_ylabel("Roll (deg)")
    axes[0].set_title("Circular Flight: Euler Angles", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, np.rad2deg(euler_hist[:, 1]), "g-", linewidth=2)
    axes[1].set_ylabel("Pitch (deg)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, np.rad2deg(euler_hist[:, 2]), "r-", linewidth=2)
    axes[2].set_ylabel("Yaw (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "circular_euler_angles.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "circular_euler_angles.pdf", bbox_inches="tight")
    plt.close()

    # Figure 4: 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos_hist[:, 1], pos_hist[:, 0], -pos_hist[:, 2], "b-", linewidth=2)
    ax.scatter(
        [pos_hist[0, 1]],
        [pos_hist[0, 0]],
        [-pos_hist[0, 2]],
        color="g",
        s=100,
        marker="o",
        label="Start",
    )
    ax.scatter(
        [pos_hist[-1, 1]],
        [pos_hist[-1, 0]],
        [-pos_hist[-1, 2]],
        color="r",
        s=100,
        marker="x",
        label="End",
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("Circular Flight: 3D Trajectory", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        results_dir / "circular_3d_trajectory.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(results_dir / "circular_3d_trajectory.pdf", bbox_inches="tight")
    plt.close()

    # Figure 5: Top-down trajectory
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(pos_hist[:, 1], pos_hist[:, 0], "b-", linewidth=2)
    ax.scatter(
        [pos_hist[0, 1]], [pos_hist[0, 0]], color="g", s=100, marker="o", label="Start"
    )
    ax.scatter(
        [pos_hist[-1, 1]], [pos_hist[-1, 0]], color="r", s=100, marker="x", label="End"
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Circular Flight: Top-Down View", fontweight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "circular_top_down.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "circular_top_down.pdf", bbox_inches="tight")
    plt.close()

    # Figure 6: Motor Commands
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for m in range(8):
        ax.plot(
            time_hist, motor_hist[:, m], color=colors[m], label=f"M{m+1}", linewidth=1.5
        )
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(-1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("Motor Command")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc="upper right", ncol=4)
    ax.set_title("Circular Flight: Motor Commands", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        results_dir / "circular_motor_commands.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(results_dir / "circular_motor_commands.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nFigures saved to {results_dir}/")

    return {
        "rms_vel_error": rms_vel_error,
        "rms_omega_error": rms_omega_error,
    }


# ============================================================================
# Test 2: Flight Path with Rate Commands
# ============================================================================
def test_flight_path():
    """Fly a path with various velocity and rate commands."""

    print("\n" + "=" * 70)
    print("TEST 2: FLIGHT PATH WITH RATE COMMANDS")
    print("=" * 70)
    print("\nVarious velocity and rate commands to test INDI tracking.")

    # Configuration
    DT = 0.01
    SIM_TIME = 30.0

    # INDI gains - MUST BE LOW to avoid saturation!
    K_vel = 0.5  # Reduced from 2.0
    K_omega = 1.0  # Reduced from 5.0
    K = np.diag([K_vel, K_vel, K_vel, K_omega, K_omega, K_omega])

    # Initialize vehicle with HIGH-FIDELITY model
    config_file = Path(__file__).parent / "omnicopter_config_hifi.yaml"
    motor_effector = MotorDynamicsEffector(num_motors=8, motor_time_constant=0.03)
    vehicle = ComplexMultirotor(str(config_file), effector=motor_effector)

    init_pos = np.array([0.0, 0.0, -3.0])
    init_vel = np.array([0.0, 0.0, 0.0])
    init_euler = np.array([0.0, 0.0, 0.0])
    init_rates = np.array([0.0, 0.0, 0.0])

    vehicle.set_initial_conditions(
        init_pos,
        init_vel,
        init_euler,
        init_rates,
        34.0,
        -86.0,
        0.0,
        np.array([20.0, 5.0, 45.0]),
    )
    vehicle.vehicle.takenoff = True

    # Initialize INDI
    indi = INDI(omit_A=True)
    indi.set_state_model(K=K, H=None, B0=B0)

    # Initial motor state - START FROM HOVER!
    # Hover scale s = 0.53 exactly balances gravity
    u_hover = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * 0.53
    u = u_hover.copy()
    motor_effector.reset(u)

    def get_reference(t):
        """Generate flight path reference.

        Timeline:
        - 0-5s: Hover, then start rolling
        - 5-10s: Roll while moving North
        - 10-15s: Stop roll, yaw 90 degrees
        - 15-20s: Move East while pitching
        - 20-25s: Descend while rolling opposite
        - 25-30s: Return to hover
        """

        # Smooth step for transitions
        def smooth(t, t0, t1):
            if t < t0:
                return 0.0
            elif t > t1:
                return 1.0
            return 0.5 * (1 - np.cos(np.pi * (t - t0) / (t1 - t0)))

        v_ned = np.zeros(3)
        omega = np.zeros(3)

        if t < 5.0:
            # Hover, start rolling at t=3
            omega[0] = np.deg2rad(30) * smooth(t, 3.0, 5.0)
        elif t < 10.0:
            # Roll 30 deg/s, move North 1 m/s
            omega[0] = np.deg2rad(30)
            v_ned[0] = 1.0 * smooth(t, 5.0, 6.0)
        elif t < 15.0:
            # Stop roll, yaw 30 deg/s, maintain North velocity then slow
            omega[0] = np.deg2rad(30) * (1 - smooth(t, 10.0, 11.0))
            omega[2] = np.deg2rad(30) * smooth(t, 10.0, 11.0)
            v_ned[0] = 1.0 * (1 - smooth(t, 13.0, 15.0))
        elif t < 20.0:
            # Move East, pitch 20 deg/s
            omega[2] = np.deg2rad(30) * (1 - smooth(t, 15.0, 16.0))
            omega[1] = np.deg2rad(20) * smooth(t, 15.0, 16.0)
            v_ned[1] = 1.0 * smooth(t, 15.0, 16.0)
        elif t < 25.0:
            # Descend, roll opposite
            omega[1] = np.deg2rad(20) * (1 - smooth(t, 20.0, 21.0))
            omega[0] = np.deg2rad(-30) * smooth(t, 20.0, 21.0)
            v_ned[1] = 1.0 * (1 - smooth(t, 23.0, 25.0))
            v_ned[2] = 0.5 * smooth(t, 20.0, 21.0) * (1 - smooth(t, 23.0, 25.0))
        else:
            # Return to hover
            omega[0] = np.deg2rad(-30) * (1 - smooth(t, 25.0, 27.0))

        return v_ned, omega

    # History
    time_hist = []
    pos_hist = []
    vel_body_hist = []
    vel_ned_hist = []
    omega_hist = []
    ref_vel_ned_hist = []
    ref_omega_hist = []
    motor_hist = []
    euler_hist = []

    prev_vel = init_vel.copy()
    prev_omega = init_rates.copy()
    prev_v_body_ref = np.zeros(3)
    prev_omega_ref = np.zeros(3)

    time = 0.0

    print(f"\nSimulating {SIM_TIME}s...")

    for i in range(int(SIM_TIME / DT)):
        # Current state
        pos = vehicle.vehicle.state[v_smap_quat.ned_pos].copy()
        vel_B = vehicle.vehicle.state[v_smap_quat.body_vel].copy()
        quat = vehicle.vehicle.state[v_smap_quat.quat].copy()
        omega_B = vehicle.vehicle.state[v_smap_quat.body_rot_rate].copy()

        # Get NED velocity for logging
        vel_ned = gmath.quat_rotate_vector(quat, vel_B)

        # Euler angles for logging
        roll, pitch, yaw = gmath.quat_to_euler(quat)

        # Get reference
        v_ned_ref, omega_ref = get_reference(time)

        # Convert NED velocity to body frame for INDI
        q_inv = gmath.quat_conjugate(quat)
        v_body_ref = gmath.quat_rotate_vector(q_inv, v_ned_ref)

        # INDI reference
        ref = np.concatenate([v_body_ref, omega_ref])

        # Reference derivatives (numerical)
        v_body_ref_dot = (v_body_ref - prev_v_body_ref) / DT
        omega_ref_dot = (omega_ref - prev_omega_ref) / DT
        ref_dot = np.concatenate([v_body_ref_dot, omega_ref_dot])

        # Current state for INDI
        x = np.concatenate([vel_B, omega_B])

        # State derivatives
        vel_dot = (vel_B - prev_vel) / DT
        omega_dot = (omega_B - prev_omega) / DT
        x_dot = np.concatenate([vel_dot, omega_dot])

        # Record history
        time_hist.append(time)
        pos_hist.append(pos.copy())
        vel_body_hist.append(vel_B.copy())
        vel_ned_hist.append(vel_ned.copy())
        omega_hist.append(omega_B.copy())
        ref_vel_ned_hist.append(v_ned_ref.copy())
        ref_omega_hist.append(omega_ref.copy())
        motor_hist.append(u.copy())
        euler_hist.append(np.array([roll, pitch, yaw]))

        # INDI control
        if i > 0:
            u = indi.calculate_control(
                cur_time=time,
                cur_state=x,
                cur_state_dot=x_dot,
                cur_input=u,
                ref=ref,
                ref_dot=ref_dot,
            )
            u = np.clip(u, -1.0, 1.0)

        prev_vel = vel_B.copy()
        prev_omega = omega_B.copy()
        prev_v_body_ref = v_body_ref.copy()
        prev_omega_ref = omega_ref.copy()

        vehicle.propagate_state(DT, vehicle.vehicle.state, u=u)
        time += DT

    # Convert to arrays
    time_hist = np.array(time_hist)
    pos_hist = np.array(pos_hist)
    vel_body_hist = np.array(vel_body_hist)
    vel_ned_hist = np.array(vel_ned_hist)
    omega_hist = np.array(omega_hist)
    ref_vel_ned_hist = np.array(ref_vel_ned_hist)
    ref_omega_hist = np.array(ref_omega_hist)
    motor_hist = np.array(motor_hist)
    euler_hist = np.array(euler_hist)

    # Compute errors
    vel_ned_error = vel_ned_hist - ref_vel_ned_hist
    omega_error = omega_hist - ref_omega_hist

    rms_vel_error = np.sqrt(np.mean(vel_ned_error**2, axis=0))
    rms_omega_error = np.sqrt(np.mean(omega_error**2, axis=0))

    print(f"\nResults:")
    print(
        f"  RMS NED velocity error: N={rms_vel_error[0]:.3f}, E={rms_vel_error[1]:.3f}, D={rms_vel_error[2]:.3f} m/s"
    )
    print(
        f"  RMS angular rate error: p={np.rad2deg(rms_omega_error[0]):.2f}, q={np.rad2deg(rms_omega_error[1]):.2f}, r={np.rad2deg(rms_omega_error[2]):.2f} deg/s"
    )

    # Save figures
    results_dir = Path(__file__).parent / "ValidationResults"
    results_dir.mkdir(exist_ok=True)

    # Figure 1: NED Velocity Tracking
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_hist, vel_ned_hist[:, 0], "b-", label="Actual", linewidth=2)
    axes[0].plot(
        time_hist,
        ref_vel_ned_hist[:, 0],
        "r--",
        label="Reference",
        linewidth=2,
        alpha=0.8,
    )
    axes[0].set_ylabel("North (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Flight Path: NED Velocity Tracking", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, vel_ned_hist[:, 1], "b-", linewidth=2)
    axes[1].plot(time_hist, ref_vel_ned_hist[:, 1], "r--", linewidth=2, alpha=0.8)
    axes[1].set_ylabel("East (m/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, vel_ned_hist[:, 2], "b-", linewidth=2)
    axes[2].plot(time_hist, ref_vel_ned_hist[:, 2], "r--", linewidth=2, alpha=0.8)
    axes[2].set_ylabel("Down (m/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        results_dir / "path_velocity_tracking.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(results_dir / "path_velocity_tracking.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: Angular Rate Tracking
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(
        time_hist, np.rad2deg(omega_hist[:, 0]), "b-", label="Actual", linewidth=2
    )
    axes[0].plot(
        time_hist,
        np.rad2deg(ref_omega_hist[:, 0]),
        "r--",
        label="Reference",
        linewidth=2,
        alpha=0.8,
    )
    axes[0].set_ylabel("p (deg/s)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Flight Path: Angular Rate Tracking", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, np.rad2deg(omega_hist[:, 1]), "b-", linewidth=2)
    axes[1].plot(
        time_hist, np.rad2deg(ref_omega_hist[:, 1]), "r--", linewidth=2, alpha=0.8
    )
    axes[1].set_ylabel("q (deg/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, np.rad2deg(omega_hist[:, 2]), "b-", linewidth=2)
    axes[2].plot(
        time_hist, np.rad2deg(ref_omega_hist[:, 2]), "r--", linewidth=2, alpha=0.8
    )
    axes[2].set_ylabel("r (deg/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "path_rate_tracking.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "path_rate_tracking.pdf", bbox_inches="tight")
    plt.close()

    # Figure 3: Euler Angles
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_hist, np.rad2deg(euler_hist[:, 0]), "b-", linewidth=2)
    axes[0].set_ylabel("Roll (deg)")
    axes[0].set_title("Flight Path: Euler Angles", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, np.rad2deg(euler_hist[:, 1]), "g-", linewidth=2)
    axes[1].set_ylabel("Pitch (deg)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, np.rad2deg(euler_hist[:, 2]), "r-", linewidth=2)
    axes[2].set_ylabel("Yaw (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "path_euler_angles.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "path_euler_angles.pdf", bbox_inches="tight")
    plt.close()

    # Figure 4: 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos_hist[:, 1], pos_hist[:, 0], -pos_hist[:, 2], "b-", linewidth=2)
    ax.scatter(
        [pos_hist[0, 1]],
        [pos_hist[0, 0]],
        [-pos_hist[0, 2]],
        color="g",
        s=100,
        marker="o",
        label="Start",
    )
    ax.scatter(
        [pos_hist[-1, 1]],
        [pos_hist[-1, 0]],
        [-pos_hist[-1, 2]],
        color="r",
        s=100,
        marker="x",
        label="End",
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("Flight Path: 3D Trajectory", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "path_3d_trajectory.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "path_3d_trajectory.pdf", bbox_inches="tight")
    plt.close()

    # Figure 5: Top-down trajectory
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(pos_hist[:, 1], pos_hist[:, 0], "b-", linewidth=2)
    ax.scatter(
        [pos_hist[0, 1]], [pos_hist[0, 0]], color="g", s=100, marker="o", label="Start"
    )
    ax.scatter(
        [pos_hist[-1, 1]], [pos_hist[-1, 0]], color="r", s=100, marker="x", label="End"
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Flight Path: Top-Down View", fontweight="bold")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "path_top_down.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "path_top_down.pdf", bbox_inches="tight")
    plt.close()

    # Figure 6: Motor Commands
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for m in range(8):
        ax.plot(
            time_hist, motor_hist[:, m], color=colors[m], label=f"M{m+1}", linewidth=1.5
        )
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(-1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("Motor Command")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-1.2, 1.2])
    ax.legend(loc="upper right", ncol=4)
    ax.set_title("Flight Path: Motor Commands", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "path_motor_commands.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "path_motor_commands.pdf", bbox_inches="tight")
    plt.close()

    # Figure 7: Position
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_hist, pos_hist[:, 0], "b-", linewidth=2)
    axes[0].set_ylabel("North (m)")
    axes[0].set_title("Flight Path: NED Position", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hist, pos_hist[:, 1], "g-", linewidth=2)
    axes[1].set_ylabel("East (m)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hist, -pos_hist[:, 2], "r-", linewidth=2)
    axes[2].set_ylabel("Altitude (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "path_position.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "path_position.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nFigures saved to {results_dir}/")

    return {
        "rms_vel_error": rms_vel_error,
        "rms_omega_error": rms_omega_error,
    }


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("OMNICOPTER INDI VALIDATION - HIGH FIDELITY MODEL")
    print("=" * 70)
    print("\nHigh-fidelity model features:")
    print("  - Quadratic thrust: T = T_max * u^2")
    print("  - Aerodynamic drag: cd = 0.3")
    print("  - Reaction torques from rotor spin")
    print("  - Motor dynamics: tau = 0.03s")
    print("\nINDI uses LINEAR B0 matrix (design model mismatch)")
    print(f"\nB0 shape: {B0.shape}")
    print(f"B0 rank: {np.linalg.matrix_rank(B0)}")
    print(f"B0 condition number: {np.linalg.cond(B0):.1f}")

    # Run Test 0 first - diagnose pure rate tracking
    results0 = test_pure_rate_tracking()

    print("\n" + "=" * 70)
    print("TEST 0 COMPLETE")
    print("=" * 70)
    print(
        f"\nTest 0 (Pure Rate Tracking): rate RMS = {np.rad2deg(np.mean(results0['rms_omega_error'])):.2f} deg/s"
    )

    # Only run other tests if rate tracking is good
    if np.rad2deg(np.mean(results0["rms_omega_error"])) < 5.0:
        print("\nRate tracking OK, running full tests...")
        results1 = test_circular_flight_with_roll()
        results2 = test_flight_path()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETE")
        print("=" * 70)
        print("\nSummary:")
        print(
            f"  Test 0 (Pure Rate): rate RMS = {np.rad2deg(np.mean(results0['rms_omega_error'])):.2f} deg/s"
        )
        print(
            f"  Test 1 (Circular + Roll): vel RMS = {np.mean(results1['rms_vel_error']):.3f} m/s, rate RMS = {np.rad2deg(np.mean(results1['rms_omega_error'])):.2f} deg/s"
        )
        print(
            f"  Test 2 (Flight Path): vel RMS = {np.mean(results2['rms_vel_error']):.3f} m/s, rate RMS = {np.rad2deg(np.mean(results2['rms_omega_error'])):.2f} deg/s"
        )
    else:
        print("\nRate tracking FAILED - skipping other tests until diagnosed")
