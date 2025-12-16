from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


NEW_METHOD_NAME = "eta"


def load_csv(path: Path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def unpack_columns(data: np.ndarray):
    t = data[:, 0]
    pn = data[:, 1:4]
    vb = data[:, 4:7]
    vb_ref = data[:, 7:10]
    omega = data[:, 10:13]
    omega_ref = data[:, 13:16]
    return t, pn, vb, vb_ref, omega, omega_ref


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


def plot_errors(test_name: str, baseline_csv: Path, eta_csv: Path, output_dir: Path):

    data_b = load_csv(baseline_csv)
    data_e = load_csv(eta_csv)

    t_b, pn_b, vb_b, vb_ref_b, om_b, om_ref_b = unpack_columns(data_b)
    t_e, pn_e, vb_e, vb_ref_e, om_e, om_ref_e = unpack_columns(data_e)

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

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.ravel()

    vb_labels = ["(a)", "(b)", "(c)"]
    for i in range(3):
        ax = axes[i]
        ax.plot(t, e_vb_base[:, i], label="baseline" if i == 0 else None)
        ax.plot(t, e_vb_eta[:, i], label=NEW_METHOD_NAME if i == 0 else None)
        ax.set_ylabel(f"{vb_labels[i]}")
        ax.grid(True, alpha=0.3)

    rate_labels = ["(d)", "(e)", "(f)"]
    for i in range(3):
        ax = axes[3 + i]
        ax.plot(t, e_om_base[:, i])
        ax.plot(t, e_om_eta[:, i])
        ax.set_ylabel(f"{rate_labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")

    axes[0].legend()
    add_panel_labels(axes)
    fig.tight_layout()
    fig.savefig(output_dir / f"{test_name}_errors_baseline_vs_eta.png", dpi=150)
    plt.close(fig)


def plot_refs(test_name: str, baseline_csv: Path, eta_csv: Path, output_dir: Path):
    data_b = load_csv(baseline_csv)
    data_e = load_csv(eta_csv)

    t_b, pn_b, vb_b, vb_ref_b, om_b, om_ref_b = unpack_columns(data_b)
    t_e, pn_e, vb_e, vb_ref_e, om_e, om_ref_e = unpack_columns(data_e)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.ravel()

    vn_labels = ["(a)", "(b)", "(c)"]
    for i in range(3):
        ax = axes[i]
        ax.plot(t_b, vb_ref_b[:, i], label="ref" if i == 0 else None)
        ax.plot(t_b, vb_b[:, i], label="baseline" if i == 0 else None)
        ax.plot(t_e, vb_e[:, i], label=NEW_METHOD_NAME if i == 0 else None)
        ax.set_ylabel(f"{vn_labels[i]}")
        ax.grid(True, alpha=0.3)

    rate_labels = ["(d)", "(e)", "(f)"]
    for i in range(3):
        ax = axes[3 + i]
        ax.plot(t_b, np.rad2deg(om_ref_b[:, i]), label="ref" if i == 0 else None)
        ax.plot(t_b, np.rad2deg(om_b[:, i]), label="baseline" if i == 0 else None)
        ax.plot(t_e, np.rad2deg(om_e[:, i]), label=NEW_METHOD_NAME if i == 0 else None)
        ax.set_ylabel(f"{rate_labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")

    axes[0].legend()
    add_panel_labels(axes)
    fig.tight_layout()
    fig.savefig(output_dir / f"{test_name}_refs_baseline_vs_eta.png", dpi=150)
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
        plot_errors(base, d["baseline"], d["eta"], results_dir)
        plot_refs(base, d["baseline"], d["eta"], results_dir)
        made_any = True

    if not made_any:
        raise FileNotFoundError("No matched baseline/eta CSV pairs found.")

    print(f"All plots saved in: {results_dir}")


if __name__ == "__main__":
    main()
