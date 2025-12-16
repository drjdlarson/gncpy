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
    return t, vb, vb_ref, omega, omega_ref


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


def plot_errors_per_test(
    test_name: str, baseline_csv: Path, eta_csv: Path, output_dir: Path
):
    data_b = load_csv(baseline_csv)
    data_e = load_csv(eta_csv)

    t_b, vb_b, vb_ref_b, om_b, om_ref_b = unpack_columns(data_b)
    t_e, vb_e, vb_ref_e, om_e, om_ref_e = unpack_columns(data_e)

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

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.ravel()

    vb_labels = ["x", "y", "z"]
    for i in range(3):
        ax = axes[i]
        ax.plot(t, e_vb_base[:, i], label="baseline" if i == 0 else None)
        ax.plot(t, e_vb_eta[:, i], label="eta" if i == 0 else None)
        ax.set_ylabel(f"$e_{{v_b,{vb_labels[i]}}}$ (m/s)")
        ax.grid(True, alpha=0.3)

    rate_labels = ["p", "q", "r"]
    for i in range(3):
        ax = axes[3 + i]
        ax.plot(t, e_om_base[:, i])
        ax.plot(t, e_om_eta[:, i])
        ax.set_ylabel(f"$e_{{\\omega,{rate_labels[i]}}}$ (deg/s)")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")

    axes[0].legend()

    add_panel_labels(axes)

    fig.tight_layout()
    fig.savefig(output_dir / f"{test_name}_errors_baseline_vs_eta.png", dpi=150)
    plt.close(fig)


def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "Final_Results"
    if not results_dir.exists():
        raise FileNotFoundError(f"{results_dir} does not exist.")

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
        plot_errors_per_test(base, d["baseline"], d["eta"], results_dir)
        made_any = True

    if not made_any:
        raise FileNotFoundError("No matched baseline/eta CSV pairs found.")

    print(f"All plots saved in: {results_dir}")


if __name__ == "__main__":
    main()
