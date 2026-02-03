from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


NEW_METHOD_NAME = "eta"


def load_csv(path: Path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def unpack_columns(data: np.ndarray):
    t = data[:, 0]
    vb = data[:, 1:4]
    vb_ref = data[:, 4:7]
    omega = data[:, 7:10]
    omega_ref = data[:, 10:13]
    motors = data[:, 16:24]
    return t, vb, vb_ref, omega, omega_ref, motors


def unpack_columns_no_degradation(data: np.ndarray):
    """Unpack columns from csv_files format (no motor degradation tests).

    Column order: t, pn_x, pn_y, pn_z, vb_x, vb_y, vb_z, vb_ref_x, vb_ref_y, vb_ref_z,
                  omega_x, omega_y, omega_z, omega_ref_x, omega_ref_y, omega_ref_z,
                  roll_deg, pitch_deg, yaw_deg, u0-u7, eta_F_0-eta_F_7, eta_M_0-eta_M_7
    """
    t = data[:, 0]
    vb = data[:, 4:7]
    vb_ref = data[:, 7:10]
    omega = data[:, 10:13]
    omega_ref = data[:, 13:16]
    motors = data[:, 19:27]
    return t, vb, vb_ref, omega, omega_ref, motors


def interp_matrix(t_src: np.ndarray, X_src: np.ndarray, t_dst: np.ndarray):
    out = np.empty((t_dst.size, X_src.shape[1]))
    for i in range(X_src.shape[1]):
        out[:, i] = np.interp(t_dst, t_src, X_src[:, i])
    return out


def common_time(t1: np.ndarray, t2: np.ndarray):
    if t1.size == t2.size and np.allclose(t1, t2, rtol=0, atol=1e-12):
        return t1
    return np.unique(np.concatenate([t1, t2]))


def add_panel_labels(axes):
    for k, ax in enumerate(axes):
        ax.text(
            0.02,
            0.95,
            f"({chr(ord('a') + k)})",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )


def plot_tracking_comparison(
    test_name: str, baseline_csv: Path, eta_csv: Path, output_dir: Path
):
    """Figure 1: Tracking comparison with references (4 subplots, 2x2)"""
    data_b = load_csv(baseline_csv)
    data_e = load_csv(eta_csv)

    t_b, vb_b, vb_ref_b, om_b, om_ref_b, _ = unpack_columns(data_b)
    t_e, vb_e, vb_ref_e, om_e, om_ref_e, _ = unpack_columns(data_e)

    # Convert omega to deg/s
    om_b = np.rad2deg(om_b)
    om_ref_b = np.rad2deg(om_ref_b)
    om_e = np.rad2deg(om_e)
    om_ref_e = np.rad2deg(om_ref_e)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)

    colors = ["r", "g", "b"]
    vb_labels = ["x", "y", "z"]
    omega_labels = ["p", "q", "r"]

    # Top-left: Baseline vb
    ax = axes[0, 0]
    for i in range(3):
        ax.plot(
            t_b,
            vb_ref_b[:, i],
            "--",
            color=colors[i],
            alpha=0.7,
            label=f"$v_{{b,{vb_labels[i]}}}$ ref",
        )
        ax.plot(
            t_b, vb_b[:, i], "-", color=colors[i], label=f"$v_{{b,{vb_labels[i]}}}$"
        )
    ax.set_ylabel("Body velocity (m/s)")
    ax.set_ylim([-1.3, 1.3])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Top-right: ne-INDI vb
    ax = axes[0, 1]
    for i in range(3):
        ax.plot(
            t_e,
            vb_ref_e[:, i],
            "--",
            color=colors[i],
            alpha=0.7,
            label=f"$v_{{b,{vb_labels[i]}}}$ ref",
        )
        ax.plot(
            t_e, vb_e[:, i], "-", color=colors[i], label=f"$v_{{b,{vb_labels[i]}}}$"
        )
    ax.set_ylabel("Body velocity (m/s)")
    ax.set_ylim([-1.3, 1.3])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Bottom-left: Baseline omega
    ax = axes[1, 0]
    for i in range(3):
        ax.plot(
            t_b,
            om_ref_b[:, i],
            "--",
            color=colors[i],
            alpha=0.7,
            label=f"$\\omega_{{{omega_labels[i]}}}$ ref",
        )
        ax.plot(
            t_b,
            om_b[:, i],
            "-",
            color=colors[i],
            label=f"$\\omega_{{{omega_labels[i]}}}$",
        )
    ax.set_ylabel("Angular rate (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-80, 80])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Bottom-right: ne-INDI omega
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(
            t_e,
            om_ref_e[:, i],
            "--",
            color=colors[i],
            alpha=0.7,
            label=f"$\\omega_{{{omega_labels[i]}}}$ ref",
        )
        ax.plot(
            t_e,
            om_e[:, i],
            "-",
            color=colors[i],
            label=f"$\\omega_{{{omega_labels[i]}}}$",
        )
    ax.set_ylabel("Angular rate (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-80, 80])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    add_panel_labels(axes.ravel())
    fig.tight_layout()
    fig.savefig(output_dir / f"{test_name}_fig1.png", dpi=150)
    plt.close(fig)


def plot_error_magnitudes(
    test_name: str, baseline_csv: Path, eta_csv: Path, output_dir: Path
):
    """Figure 2: Error magnitudes (4 subplots, 2x2)"""
    data_b = load_csv(baseline_csv)
    data_e = load_csv(eta_csv)

    t_b, vb_b, vb_ref_b, om_b, om_ref_b, _ = unpack_columns(data_b)
    t_e, vb_e, vb_ref_e, om_e, om_ref_e, _ = unpack_columns(data_e)

    t = common_time(t_b, t_e)

    if t.size != t_b.size or not np.allclose(t, t_b, rtol=0, atol=1e-12):
        vb_b = interp_matrix(t_b, vb_b, t)
        vb_ref_b = interp_matrix(t_b, vb_ref_b, t)
        om_b = interp_matrix(t_b, om_b, t)
        om_ref_b = interp_matrix(t_b, om_ref_b, t)

    if t.size != t_e.size or not np.allclose(t, t_e, rtol=0, atol=1e-12):
        vb_e = interp_matrix(t_e, vb_e, t)
        vb_ref_e = interp_matrix(t_e, vb_ref_e, t)
        om_e = interp_matrix(t_e, om_e, t)
        om_ref_e = interp_matrix(t_e, om_ref_e, t)

    e_vb_base = np.abs(vb_b - vb_ref_b)
    e_vb_eta = np.abs(vb_e - vb_ref_e)
    e_om_base = np.abs(np.rad2deg(om_b - om_ref_b))
    e_om_eta = np.abs(np.rad2deg(om_e - om_ref_e))

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)

    colors = ["r", "g", "b"]
    vb_labels = ["x", "y", "z"]
    omega_labels = ["p", "q", "r"]

    # Top-left: Baseline vb errors
    ax = axes[0, 0]
    for i in range(3):
        ax.plot(
            t,
            e_vb_base[:, i],
            "-",
            color=colors[i],
            label=f"$e_{{v_{{b,{vb_labels[i]}}}}}$",
        )
    ax.set_ylabel("Velocity error (m/s)")
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Top-right: ne-INDI vb errors
    ax = axes[0, 1]
    for i in range(3):
        ax.plot(
            t,
            e_vb_eta[:, i],
            "-",
            color=colors[i],
            label=f"$e_{{v_{{b,{vb_labels[i]}}}}}$",
        )
    ax.set_ylabel("Velocity error (m/s)")
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Bottom-left: Baseline omega errors
    ax = axes[1, 0]
    for i in range(3):
        ax.plot(
            t,
            e_om_base[:, i],
            "-",
            color=colors[i],
            label=f"$e_{{\\omega_{{{omega_labels[i]}}}}}$",
        )
    ax.set_ylabel("Angular rate error (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-85, 85])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Bottom-right: ne-INDI omega errors
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(
            t,
            e_om_eta[:, i],
            "-",
            color=colors[i],
            label=f"$e_{{\\omega_{{{omega_labels[i]}}}}}$",
        )
    ax.set_ylabel("Angular rate error (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-85, 85])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    add_panel_labels(axes.ravel())
    fig.tight_layout()
    fig.savefig(output_dir / f"{test_name}_fig2.png", dpi=150)
    plt.close(fig)


def plot_motor_commands(
    test_name: str, baseline_csv: Path, eta_csv: Path, output_dir: Path
):
    """Figure 3: Motor commands (2 subplots, 1x2)"""
    data_b = load_csv(baseline_csv)
    data_e = load_csv(eta_csv)

    t_b, _, _, _, _, motors_b = unpack_columns(data_b)
    t_e, _, _, _, _, motors_e = unpack_columns(data_e)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Left: Baseline motors
    ax = axes[0]
    for i in range(8):
        ax.plot(t_b, motors_b[:, i], "-", label=f"U{i+1}")
    ax.set_ylabel("Motor command")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-1, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    # Right: ne-INDI motors
    ax = axes[1]
    for i in range(8):
        ax.plot(t_e, motors_e[:, i], "-", label=f"U{i+1}")
    ax.set_ylabel("Motor command")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-1, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    add_panel_labels(axes)
    fig.tight_layout()
    fig.savefig(output_dir / f"{test_name}_fig3.png", dpi=150)
    plt.close(fig)


def plot_tracking_single_test(test_name: str, csv_path: Path, output_dir: Path):
    """Figure 1 for single test (no degradation): Tracking with references (2 subplots, 1x2)"""
    data = load_csv(csv_path)
    t, vb, vb_ref, om, om_ref, _ = unpack_columns_no_degradation(data)

    # Convert omega to deg/s
    om = np.rad2deg(om)
    om_ref = np.rad2deg(om_ref)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    colors = ["r", "g", "b"]
    vb_labels = ["x", "y", "z"]
    omega_labels = ["p", "q", "r"]

    # Left: vb tracking
    ax = axes[0]
    for i in range(3):
        ax.plot(
            t,
            vb_ref[:, i],
            "--",
            color=colors[i],
            alpha=0.7,
            label=f"$v_{{b,{vb_labels[i]}}}$ ref",
        )
        ax.plot(t, vb[:, i], "-", color=colors[i], label=f"$v_{{b,{vb_labels[i]}}}$")
    ax.set_ylabel("Body velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-1.3, 1.3])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Right: omega tracking
    ax = axes[1]
    for i in range(3):
        ax.plot(
            t,
            om_ref[:, i],
            "--",
            color=colors[i],
            alpha=0.7,
            label=f"$\\omega_{{{omega_labels[i]}}}$ ref",
        )
        ax.plot(
            t, om[:, i], "-", color=colors[i], label=f"$\\omega_{{{omega_labels[i]}}}$"
        )
    ax.set_ylabel("Angular rate (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-80, 80])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    add_panel_labels(axes)
    fig.tight_layout()
    fig.savefig(output_dir / f"{test_name}_no_degradation_fig1.png", dpi=150)
    plt.close(fig)


def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "Final_Results"
    csv_files_dir = base_dir / "csv_files"
    output_dir = base_dir / "pretty_data"

    if not results_dir.exists():
        raise FileNotFoundError(f"{results_dir} does not exist.")

    output_dir.mkdir(exist_ok=True)

    # Process Final_Results (motor degradation tests: baseline vs eta)
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    pat = re.compile(r"^(?P<base>.+?)_(?P<ctrl>baseline|eta)$", re.IGNORECASE)
    groups = {}

    for p in csv_files:
        m = pat.match(p.stem)
        if not m:
            continue
        base = m.group("base")
        ctrl = m.group("ctrl").lower()
        groups.setdefault(base, {})[ctrl] = p

    made_any = False
    for base, d in sorted(groups.items()):
        if "baseline" not in d or "eta" not in d:
            continue
        print(f"Plotting {base} (with motor degradation)...")
        plot_tracking_comparison(base, d["baseline"], d["eta"], output_dir)
        plot_error_magnitudes(base, d["baseline"], d["eta"], output_dir)
        plot_motor_commands(base, d["baseline"], d["eta"], output_dir)
        made_any = True

    if not made_any:
        raise FileNotFoundError("No matched baseline/eta CSV pairs found.")

    # Process csv_files (no motor degradation tests: single INDI runs)
    if csv_files_dir.exists():
        single_csv_files = sorted(csv_files_dir.glob("indi_test*.csv"))
        if single_csv_files:
            print("\nProcessing tests without motor degradation...")
            for csv_path in single_csv_files:
                # Extract test name from filename (e.g., "indi_test1_circular_velocity_wind.csv" -> "test1")
                stem = csv_path.stem
                if stem.startswith("indi_test"):
                    test_num = stem.split("_")[1]  # Extract "test1", "test2", etc.
                    print(f"Plotting {test_num} (no motor degradation)...")
                    plot_tracking_single_test(test_num, csv_path, output_dir)

    print(f"\nAll plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
