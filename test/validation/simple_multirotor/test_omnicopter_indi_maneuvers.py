"""Baseline omnicopter INDI maneuver validation.

This is the main baseline script for the omnicopter. It keeps a small set of
named reference-tracking cases and relies on shared helpers for setup, sensing,
simulation, and plotting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from omnicopter_test_utils import (
    DEFAULT_SENSOR_CONFIG,
    circular_velocity_reference,
    combined_velocity_roll_reference,
    create_omnicopter_setup,
    helix_tumble_reference,
    plot_rate_tracking_results,
    print_case_summary,
    print_setup_summary,
    roll_rate_reference,
    run_indi_rate_case,
)

USE_IDEAL_BASELINE = False
BASELINE_MOTOR_EFFICIENCY = np.array([0.99, 1.01, 0.98, 1.02, 1.01, 0.99, 1.00, 0.98])
BASELINE_SENSOR_CONFIG = DEFAULT_SENSOR_CONFIG


def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "ValidationResults"

    setup = create_omnicopter_setup(
        base_dir / "omnicopter_config.yaml",
        base_dir / "omnicopter_config_hifi.yaml",
        motor_efficiency=BASELINE_MOTOR_EFFICIENCY,
        efficiency_mode="none",
        use_motor_dynamics=not USE_IDEAL_BASELINE,
    )

    print_setup_summary(setup)
    if USE_IDEAL_BASELINE:
        print("Baseline mode: idealized INDI benchmark")
        print("Motor mismatch: disabled")
        print("Motor dynamics: disabled")
        print("Measurement noise/filter lag: disabled")
    else:
        print(f"Baseline motor efficiency mismatch: {BASELINE_MOTOR_EFFICIENCY}")

    cases = [
        {
            "banner": "TEST 1: Circular Velocity Tracking",
            "name": "Test 1: Circular Velocity Tracking",
            "sim_time": 10.0,
            "reference": circular_velocity_reference(speed=1.0, period=8.0),
            "plot_title": "Test 1: Circular Velocity Tracking (Zero Angular Rates)",
            "output_file": output_dir / "indi_test1_circular_velocity.png",
            "seed": 101,
            "notes": [
                "Circle velocity: 1.0 m/s",
                "Circle period: 8.00 s",
            ],
        },
        {
            "banner": "TEST 2: 360 Degree Roll Maneuver",
            "name": "Test 2: 360 Degree Roll Maneuver",
            "sim_time": 10.0,
            "reference": roll_rate_reference(rate=2.0 * np.pi / 8.0, duration=8.0),
            "plot_title": "Test 2: 360 Degree Roll Maneuver (Zero Velocity)",
            "output_file": output_dir / "indi_test2_360_roll.png",
            "seed": 102,
            "notes": [
                "Roll maneuver time: 8.0 s",
                "Roll rate: 45.00 deg/s",
            ],
        },
        {
            "banner": "TEST 3: Combined Velocity + Attitude Maneuver",
            "name": "Test 3: Combined Velocity + Attitude Maneuver",
            "sim_time": 12.0,
            "reference": combined_velocity_roll_reference(
                speed=0.8,
                period=10.0,
                roll_rate=2.0 * np.pi / 8.0,
                roll_duration=8.0,
            ),
            "plot_title": "Test 3: Combined Velocity + Attitude Maneuver",
            "output_file": output_dir / "indi_test3_combined.png",
            "seed": 103,
            "notes": [
                "Circle velocity: 0.8 m/s",
                "Circle period: 10.00 s",
                "Roll maneuver time: 8.0 s",
                "Roll rate: 45.00 deg/s",
            ],
        },
        {
            "banner": "TEST 4: Helix Trajectory with 3-Axis Tumble",
            "name": "Test 4: Helix Trajectory with 3-Axis Tumble",
            "sim_time": 12.0,
            "reference": helix_tumble_reference(
                circle_speed=0.6,
                circle_period=10.0,
                vertical_speed_amp=0.3,
                vertical_period=6.0,
                roll_rate=2.0 * np.pi / 8.0,
                pitch_rate=2.0 * np.pi / 10.0,
                yaw_rate=2.0 * np.pi / 12.0,
                tumble_duration=10.0,
            ),
            "plot_title": "Test 4: Helix Trajectory with 3-Axis Tumble",
            "output_file": output_dir / "indi_test4_helix_tumble.png",
            "seed": 104,
            "notes": [
                "Helix horizontal circle: 0.6 m/s, period: 10.0 s",
                "Helix vertical oscillation: ±0.3 m/s, period: 6.0 s",
                "Tumble roll rate: 45.0 deg/s",
                "Tumble pitch rate: 36.0 deg/s",
                "Tumble yaw rate: 30.0 deg/s",
            ],
        },
    ]

    for case in cases:
        print("\n" + "=" * 70)
        print(case["banner"])
        print("=" * 70)
        for line in case["notes"]:
            print(line)
        print(f"Simulation time: {case['sim_time']:.1f} s")

        results = run_indi_rate_case(
            setup=setup,
            test_name=case["name"],
            sim_time=case["sim_time"],
            reference_func=case["reference"],
            sensor_config=BASELINE_SENSOR_CONFIG,
            sensor_seed=case["seed"],
            perfect_sensors=USE_IDEAL_BASELINE,
        )

        print_case_summary(results)
        print(
            f"Final velocity error: "
            f"{np.linalg.norm(results['body_vel'][-1, :] - results['body_vel_ref'][-1, :]):.4f} m/s"
        )
        print(
            f"Final angular rate: {np.linalg.norm(results['body_omega'][-1, :]):.6f} rad/s"
        )

        if "Roll Maneuver" in case["name"] or "Attitude" in case["name"]:
            print(f"Final roll angle: {results['euler'][-1, 0]:.2f} deg")
        if "Helix" in case["name"]:
            altitude_delta = results["ned_pos"][-1, 2] - results["ned_pos"][0, 2]
            altitude_span = (results["ned_pos"][:, 2] - results["ned_pos"][0, 2]).max()
            print(f"Final altitude change: {altitude_delta:.2f} m")
            print(f"Max altitude change: {altitude_span:.2f} m")

        plot_rate_tracking_results(results, case["plot_title"], case["output_file"])

    plt.show()


if __name__ == "__main__":
    main()
