"""Baseline omnicopter INDI hover validation.

This script is the compact sanity-check for the omnicopter stack:
- build the LoFi control effectiveness matrix used by INDI
- compute the HiFi hover trim
- verify a one-step hover propagation
- close the loop on a lightly perturbed hover condition
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from omnicopter_test_utils import (
    DT,
    INITIAL_ANGULAR_VELOCITY,
    INITIAL_ATTITUDE,
    INITIAL_POSITION,
    INITIAL_VELOCITY,
    create_omnicopter_setup,
    plot_hover_results,
    print_case_summary,
    print_setup_summary,
    reset_hifi_vehicle,
    run_indi_rate_case,
)
from gncpy.dynamics.aircraft.complex_multirotor import v_smap_quat


PERTURBED_VELOCITY = np.array([0.1, -0.05, 0.08])
PERTURBED_OMEGA = np.array([0.02, -0.03, 0.01])
SIM_TIME_HOVER = 3.0


def zero_reference(_):
    vb_ref = np.zeros(3)
    omega_ref = np.zeros(3)
    vb_ref_dot = np.zeros(3)
    omega_ref_dot = np.zeros(3)
    return vb_ref, omega_ref, vb_ref_dot, omega_ref_dot


def main():
    base_dir = Path(__file__).parent
    setup = create_omnicopter_setup(
        base_dir / "omnicopter_config.yaml",
        base_dir / "omnicopter_config_hifi.yaml",
    )

    print_setup_summary(setup)
    print(f"B0 rank: {np.linalg.matrix_rank(setup['b0'])}")
    print(f"Max |hover command|: {np.max(np.abs(setup['hover_cmds_hifi'])):.4f}")

    reset_hifi_vehicle(setup)
    one_step_state = setup["hifi_dyn"].propagate_state(
        DT,
        setup["hifi_dyn"].vehicle.state,
        setup["hover_cmds_hifi"],
    ).flatten()
    one_step_vel = one_step_state[v_smap_quat.body_vel].flatten()
    one_step_omega = one_step_state[v_smap_quat.body_rot_rate].flatten()
    print(f"One-step hover body velocity: {one_step_vel}")
    print(f"One-step hover body rate: {one_step_omega}")

    print("\n" + "=" * 60)
    print("Simulating INDI hover recovery")
    print("=" * 60)
    print(f"Initial velocity perturbation: {PERTURBED_VELOCITY} m/s")
    print(f"Initial angular-rate perturbation: {np.rad2deg(PERTURBED_OMEGA)} deg/s")

    results = run_indi_rate_case(
        setup=setup,
        test_name="Hover Recovery",
        sim_time=SIM_TIME_HOVER,
        reference_func=zero_reference,
        initial_position=INITIAL_POSITION,
        initial_velocity=PERTURBED_VELOCITY,
        initial_attitude=INITIAL_ATTITUDE,
        initial_angular_velocity=PERTURBED_OMEGA,
    )

    print_case_summary(results)
    print(f"Final body velocity: {results['body_vel'][-1, :]}")
    print(f"Final body angular rate: {results['body_omega'][-1, :]}")
    print(f"Final NED position: {results['ned_pos'][-1, :]}")

    plot_hover_results(
        results,
        title="INDI Hover Control - HiFi Model with Motor Dynamics",
        output_file=base_dir / "ValidationResults" / "indi_hover_test.png",
        depth_ref=INITIAL_POSITION[2],
    )
    plt.show()


if __name__ == "__main__":
    main()
