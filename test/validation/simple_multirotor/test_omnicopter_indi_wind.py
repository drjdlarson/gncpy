"""Omnicopter INDI robustness study with wind and degraded motors.

This script is intentionally *not* the clean baseline. It layers wind
disturbance and motor-effectiveness loss on top of the same maneuver set used
by the baseline script so the comparison is easy to interpret.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from omnicopter_test_utils import (
    circular_velocity_reference,
    combined_velocity_roll_reference,
    create_omnicopter_setup,
    helix_tumble_reference,
    make_wind_disturbance,
    plot_rate_tracking_results,
    print_case_summary,
    print_setup_summary,
    roll_rate_reference,
    run_indi_rate_case,
)


DEGRADED_MOTOR_EFFICIENCY = np.array([0.69, 1.01, 0.98, 1.02, 1.01, 0.67, 1.00, 0.98])
WIND_VELOCITY = np.array([8.0, 5.0, 1.0])
WIND_GUST_AMP = np.array([4.0, 3.0, 1.0])
WIND_GUST_FREQ = np.array([0.3, 0.4, 0.5])


def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "ValidationResults"

    setup = create_omnicopter_setup(
        base_dir / "omnicopter_config.yaml",
        base_dir / "omnicopter_config_hifi.yaml",
        motor_efficiency=DEGRADED_MOTOR_EFFICIENCY,
        efficiency_mode="output",
    )
    disturbance = make_wind_disturbance(
        wind_velocity=WIND_VELOCITY,
        wind_gust_amp=WIND_GUST_AMP,
        wind_gust_freq=WIND_GUST_FREQ,
        update_derived_states=True,
    )

    print_setup_summary(setup)
    print("Robustness scenario: wind disturbance plus degraded motors")
    print(f"Motor effectiveness: {DEGRADED_MOTOR_EFFICIENCY}")
    print(f"Constant wind (NED): {WIND_VELOCITY}")
    print(f"Wind gust amplitude: {WIND_GUST_AMP}")
    print(f"Wind gust frequencies (Hz): {WIND_GUST_FREQ}")

    cases = [
        {
            "banner": "TEST 1: Circular Velocity Tracking (wind + degraded motors)",
            "name": "Test 1: Circular Velocity Tracking",
            "sim_time": 10.0,
            "reference": circular_velocity_reference(speed=1.0, period=8.0),
            "plot_title": "Test 1: Circular Velocity Tracking (Wind + Degraded Motors)",
            "output_file": output_dir / "indi_test1_circular_velocity_wind.png",
            "seed": 201,
        },
        {
            "banner": "TEST 2: 360 Degree Roll Maneuver (wind + degraded motors)",
            "name": "Test 2: 360 Degree Roll Maneuver",
            "sim_time": 10.0,
            "reference": roll_rate_reference(rate=2.0 * np.pi / 8.0, duration=8.0),
            "plot_title": "Test 2: 360 Degree Roll Maneuver (Wind + Degraded Motors)",
            "output_file": output_dir / "indi_test2_360_roll_wind.png",
            "seed": 202,
        },
        {
            "banner": "TEST 3: Combined Velocity + Attitude Maneuver (wind + degraded motors)",
            "name": "Test 3: Combined Velocity + Attitude Maneuver",
            "sim_time": 12.0,
            "reference": combined_velocity_roll_reference(
                speed=0.8,
                period=10.0,
                roll_rate=2.0 * np.pi / 8.0,
                roll_duration=8.0,
            ),
            "plot_title": "Test 3: Combined Velocity + Attitude Maneuver (Wind + Degraded Motors)",
            "output_file": output_dir / "indi_test3_combined_wind.png",
            "seed": 203,
        },
        {
            "banner": "TEST 4: Helix Trajectory with 3-Axis Tumble (wind + degraded motors)",
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
            "plot_title": "Test 4: Helix Trajectory with 3-Axis Tumble (Wind + Degraded Motors)",
            "output_file": output_dir / "indi_test4_helix_tumble_wind.png",
            "seed": 204,
        },
    ]

    for case in cases:
        print("\n" + "=" * 70)
        print(case["banner"])
        print("=" * 70)
        print(f"Simulation time: {case['sim_time']:.1f} s")

        results = run_indi_rate_case(
            setup=setup,
            test_name=case["name"],
            sim_time=case["sim_time"],
            reference_func=case["reference"],
            sensor_seed=case["seed"],
            disturbance_callback=disturbance,
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
            altitude_span = (
                results["ned_pos"][:, 2] - results["ned_pos"][0, 2]
            ).max()
            print(f"Final altitude change: {altitude_delta:.2f} m")
            print(f"Max altitude change: {altitude_span:.2f} m")

        plot_rate_tracking_results(results, case["plot_title"], case["output_file"])

    plt.show()


if __name__ == "__main__":
    main()
