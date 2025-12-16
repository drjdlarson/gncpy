from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


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

    vb_b = interp_matrix(t_b, vb_b, t)
    vb_ref_b = interp_matrix(t_b, vb_ref_b, t)
    om_b = interp_matrix(t_b, om_b, t)
    om_ref_b = interp_matrix(t_b, om_ref_b, t)

    vb_e = interp_matrix(t_e, vb_e, t)
    vb_ref_e = interp_matrix(t_e, vb_ref_e, t)
    om_e = interp_matrix(t_e, om_e, t)
    om_ref_e = interp_matrix(t_e, om_ref_e, t)

    e_vb_base = vb_b - vb_ref_b
    e_vb_eta = vb_e - vb_ref_e
    e_om_base = np.rad2deg(om_b - om_ref_b)
    e_om_eta = np.rad2deg(om_e - om_ref_e)

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


def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "Final_Results"
    output_dir = base_dir / "pretty_data"

    if not results_dir.exists():
        raise FileNotFoundError(f"{results_dir} does not exist.")

    output_dir.mkdir(exist_ok=True)

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
        print(f"Plotting {base}...")
        plot_tracking_comparison(base, d["baseline"], d["eta"], output_dir)
        plot_error_magnitudes(base, d["baseline"], d["eta"], output_dir)
        plot_motor_commands(base, d["baseline"], d["eta"], output_dir)
        made_any = True

    if not made_any:
        raise FileNotFoundError("No matched baseline/eta CSV pairs found.")

    print(f"All plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
