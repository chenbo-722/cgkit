"""plot-test domain logic: compare DeepMD model predictions vs reference data.

After training a DeepMD potential, the ``dp test`` command produces prediction
files (``energy.raw``, ``force.raw``) that can be compared against the
reference data in the system directory.  This module reads both and renders
publication-quality comparison plots:

- Force parity scatter (predicted vs reference, per-component subplots)
- Energy parity scatter (predicted vs reference)
- Force error distribution histogram
- Energy error distribution histogram
- Summary statistics table (RMSE, MAE, R²)

Design: matplotlib is imported lazily so ``cgkit.py`` stays lightweight and
all non-plotting subcommands work without matplotlib installed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Matplotlib is deferred — populated by _import_deps().
plt = None
mcolors = None


def _import_deps() -> None:
    """Populate matplotlib module globals. Idempotent."""
    global plt, mcolors
    if plt is not None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _plt.style.use("seaborn-v0_8-whitegrid")
    plt = _plt
    import matplotlib.colors as _mcolors
    mcolors = _mcolors


# =============================================================================
# I/O helpers
# =============================================================================

def _read_raw(filepath: str) -> np.ndarray:
    """Read a DeepMD ``.raw`` file (space-separated values).

    Returns:
        2-d array of shape ``(n_frames, n_values_per_frame)``.
        For ``energy.raw`` this is ``(n_frames, 1)``;
        for ``force.raw`` it is ``(n_frames, n_atoms * 3)``.
    """
    data: list[list[float]] = []
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            data.append([float(x) for x in stripped.split()])
    return np.array(data) if data else np.array([[]])


def _read_npy(filepath: str) -> np.ndarray:
    """Read a DeepMD ``.npy`` file.

    Returns:
        numpy array as stored on disk (typically 2-d for forces,
        1-d or 2-d for energies).
    """
    arr = np.load(filepath)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _find_data_file(dir_path: str, basename: str) -> str:
    """Find a data file by basename in *dir_path*, trying ``.npy`` then ``.raw``.

    Returns the full path, or an empty string if neither exists.
    """
    for suffix in (".npy", ".raw"):
        full = os.path.join(dir_path, basename + suffix)
        if os.path.isfile(full):
            return full
    return ""


def _read_data(dir_path: str, basename: str) -> np.ndarray:
    """Read data from *dir_path* for the given *basename* (auto-detects format)."""
    path = _find_data_file(dir_path, basename)
    if not path:
        raise FileNotFoundError(
            f"Neither {basename}.raw nor {basename}.npy found in {dir_path}"
        )
    if path.endswith(".npy"):
        return _read_npy(path)
    return _read_raw(path)


# =============================================================================
# Metrics
# =============================================================================

def _compute_metrics(pred: np.ndarray, ref: np.ndarray,
                     label: str = "") -> Dict[str, float]:
    """Compute regression metrics between predicted and reference arrays.

    Args:
        pred: Predicted values (1-d or 2-d).
        ref: Reference values (same shape as *pred*).
        label: Human-readable label for printing.

    Returns:
        Dict with keys ``rmse``, ``mae``, ``r2``.
    """
    pred_f = pred.flatten()
    ref_f = ref.flatten()
    mask = np.isfinite(pred_f) & np.isfinite(ref_f)
    pred_f, ref_f = pred_f[mask], ref_f[mask]

    diff = pred_f - ref_f
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((ref_f - ref_f.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2}


# =============================================================================
# Plots
# =============================================================================

def _force_parity_plot(pred_forces: np.ndarray, ref_forces: np.ndarray,
                       output_dir: str) -> None:
    """Render per-component force parity subplots (Fx, Fy, Fz).

    Each subplot shows a scatter of predicted vs reference force components
    with the y=x identity line plus RMSE / R² annotations.
    """
    _import_deps()

    n_atoms = pred_forces.shape[1] // 3
    if n_atoms == 0:
        print("[plot-test] WARNING: empty force data; skipping force parity plot")
        return

    # Reshape to (n_frames * n_atoms, 3) for per-component access.
    pred_3d = pred_forces.reshape(-1, 3)
    ref_3d = ref_forces.reshape(-1, 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    labels = [r"$F_x$", r"$F_y$", r"$F_z$"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (ax, lbl, c) in enumerate(zip(axes, labels, colors)):
        p = pred_3d[:, i]
        r = ref_3d[:, i]

        # Down-sample for scatter if very large
        n = len(p)
        show = slice(None) if n <= 20000 else np.random.choice(n, size=20000, replace=False)
        ax.scatter(r[show], p[show], alpha=0.25, s=3, c=c, edgecolors="none")

        # Identity line (y = x)
        lims = (min(r.min(), p.min()), max(r.max(), p.max()))
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        metrics = _compute_metrics(p, r)
        ax.set_title(f"{lbl}\nRMSE={metrics['rmse']:.4f}   "
                     f"MAE={metrics['mae']:.4f}   R²={metrics['r2']:.4f}",
                     fontsize=10)
        ax.set_xlabel("Reference Force (eV/Å)")
        ax.set_ylabel("Predicted Force (eV/Å)")

    fig.suptitle("Force Predictions vs Reference", fontsize=13, y=1.01)
    fig.tight_layout()

    out = os.path.join(output_dir, "force_parity.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _energy_parity_plot(pred_energy: np.ndarray, ref_energy: np.ndarray,
                        output_dir: str) -> None:
    """Scatter of predicted energy vs reference energy with identity line."""
    _import_deps()

    p = pred_energy.flatten()
    r = ref_energy.flatten()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r, p, alpha=0.5, s=20, c="#1f77b4", edgecolors="none")

    lims = (min(r.min(), p.min()), max(r.max(), p.max()))
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    metrics = _compute_metrics(p, r)
    ax.set_title(f"Energy Predictions vs Reference\n"
                 f"RMSE = {metrics['rmse']:.6g}   MAE = {metrics['mae']:.6g}   "
                 f"R² = {metrics['r2']:.6g}",
                 fontsize=11)
    ax.set_xlabel("Reference Energy (eV)")
    ax.set_ylabel("Predicted Energy (eV)")

    fig.tight_layout()
    out = os.path.join(output_dir, "energy_parity.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _force_error_hist(pred_forces: np.ndarray, ref_forces: np.ndarray,
                      output_dir: str) -> None:
    """Histogram of force prediction errors (pred - ref)."""
    _import_deps()

    errors = (pred_forces.flatten() - ref_forces.flatten())
    errors = errors[np.isfinite(errors)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=80, color="#1f77b4", alpha=0.75, edgecolor="white",
            linewidth=0.3)

    metrics = _compute_metrics(pred_forces, ref_forces, "Force")
    std = np.std(errors)
    ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.4)
    ax.axvline(-std, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.axvline(+std, color="gray", ls=":", lw=0.6, alpha=0.5)

    ax.set_xlabel("Force Error (eV/Å): $F_{pred} - F_{ref}$")
    ax.set_ylabel("Count")
    ax.set_title(f"Force Error Distribution  "
                 f"(μ = {errors.mean():.4f}, σ = {std:.4f}, "
                 f"RMSE = {metrics['rmse']:.4f})",
                 fontsize=11)
    fig.tight_layout()

    out = os.path.join(output_dir, "force_error_dist.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _energy_error_hist(pred_energy: np.ndarray, ref_energy: np.ndarray,
                       output_dir: str) -> None:
    """Histogram of energy prediction errors (pred - ref)."""
    _import_deps()

    errors = (pred_energy.flatten() - ref_energy.flatten())
    errors = errors[np.isfinite(errors)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, color="#2ca02c", alpha=0.75, edgecolor="white",
            linewidth=0.3)

    metrics = _compute_metrics(pred_energy, ref_energy, "Energy")
    std = np.std(errors)
    ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.4)
    ax.axvline(-std, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.axvline(+std, color="gray", ls=":", lw=0.6, alpha=0.5)

    ax.set_xlabel("Energy Error (eV): $E_{pred} - E_{ref}$")
    ax.set_ylabel("Count")
    ax.set_title(f"Energy Error Distribution  "
                 f"(μ = {errors.mean():.6g}, σ = {std:.6g}, "
                 f"RMSE = {metrics['rmse']:.6g})",
                 fontsize=11)
    fig.tight_layout()

    out = os.path.join(output_dir, "energy_error_dist.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Entry point
# =============================================================================

def run(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """cgkit plot-test entry point.

    Reads model predictions and reference data, then generates comparison
    plots and a summary CSV.

    Data sources are resolved in order:
    1. ``--pred-dir`` / ``--ref-dir`` CLI args.
    2. ``plot_test.pred_dir`` / ``plot_test.ref_dir`` from config.json.
    3. Auto-detect: look for a ``<pred_dir>/../<system>`` sibling.
    """
    plot_cfg = config.get("plot_test", {}) or {}
    paths = config.get("paths", {})

    # --- Resolve prediction directory ---
    pred_dir = (
        getattr(args, "pred_dir", None)
        or plot_cfg.get("pred_dir")
        or paths.get("deepmd_output_base_dir")
    )
    if not pred_dir:
        print("[plot-test] ERROR: --pred-dir is required (no config default found)")
        return 1
    pred_dir = os.path.abspath(os.path.expanduser(pred_dir))

    # --- Resolve reference (system) directory ---
    ref_dir = (
        getattr(args, "ref_dir", None)
        or plot_cfg.get("ref_dir")
    )
    if not ref_dir:
        # Heuristic: look for a sibling directory named with system subdir
        # (common dp test layout: <set>/test/<system>/  with predictions in
        # that dir and reference in a parent-level system dir.)
        parent = os.path.dirname(pred_dir)
        grandparent = os.path.dirname(parent)
        for candidate in (
            os.path.join(parent, "..", "set.000"),
            grandparent,
            parent,
        ):
            cd = os.path.normpath(os.path.join(pred_dir, candidate))
            if _find_data_file(cd, "force"):
                ref_dir = os.path.realpath(cd)
                break
    if not ref_dir:
        print("[plot-test] ERROR: --ref-dir is required (could not auto-detect)")
        return 1
    ref_dir = os.path.abspath(os.path.expanduser(ref_dir))

    # --- Resolve output directory ---
    output_dir = (
        getattr(args, "output_dir", None)
        or plot_cfg.get("output_dir")
        or paths.get("analysis_output_base_dir")
    )
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(pred_dir), "test_comparison")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # --- Max frames ---
    max_frames = getattr(args, "max_frames", None) or plot_cfg.get("max_frames")

    # --- Skip plots ---
    skip = getattr(args, "skip", None)
    skip_plots = set(skip) if skip else set(plot_cfg.get("skip_plots", []))

    print(f"[plot-test] pred_dir   = {pred_dir}")
    print(f"[plot-test] ref_dir    = {ref_dir}")
    print(f"[plot-test] output_dir = {output_dir}")

    # ------------------------------------------------------------------
    # Read data
    # ------------------------------------------------------------------
    try:
        pred_energy = _read_data(pred_dir, "energy")
        ref_energy = _read_data(ref_dir, "energy")
        print(f"[plot-test] energy:   {pred_energy.shape[0]} pred vs "
              f"{ref_energy.shape[0]} ref frames")
    except FileNotFoundError as e:
        print(f"[plot-test] WARNING: energy data missing — {e}")
        pred_energy = np.array([[]])
        ref_energy = np.array([[]])

    try:
        pred_forces = _read_data(pred_dir, "force")
        ref_forces = _read_data(ref_dir, "force")
        print(f"[plot-test] forces:   {pred_forces.shape[0]} pred vs "
              f"{ref_forces.shape[0]} ref frames "
              f"({pred_forces.shape[1] // 3} atoms)")
    except FileNotFoundError as e:
        print(f"[plot-test] WARNING: force data missing — {e}")
        pred_forces = np.array([[]])
        ref_forces = np.array([[]])

    # --- Frame alignment ---
    n_frames = min(pred_forces.shape[0], ref_forces.shape[0]) if pred_forces.size else 0
    n_e_frames = min(pred_energy.shape[0], ref_energy.shape[0]) if pred_energy.size else 0

    if max_frames and n_frames > max_frames:
        idx = np.linspace(0, n_frames - 1, num=max_frames, dtype=int)
        pred_forces = pred_forces[idx]
        ref_forces = ref_forces[idx]
        n_frames = max_frames
        print(f"[plot-test] forces down-sampled to {max_frames} frames")

    if max_frames and n_e_frames > max_frames:
        idx = np.linspace(0, n_e_frames - 1, num=max_frames, dtype=int)
        pred_energy = pred_energy[idx]
        ref_energy = ref_energy[idx]
        n_e_frames = max_frames
        print(f"[plot-test] energy down-sampled to {max_frames} frames")

    if n_frames == 0 and n_e_frames == 0:
        print("[plot-test] ERROR: no data to compare")
        return 1

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  MODEL PREDICTION vs REFERENCE — SUMMARY")
    print("=" * 60)

    records: list[Dict[str, Any]] = []

    if pred_forces.size and ref_forces.size:
        metrics_f = _compute_metrics(pred_forces, ref_forces, "Force")
        print(f"  Force   RMSE = {metrics_f['rmse']:.6f} eV/Å")
        print(f"  Force   MAE  = {metrics_f['mae']:.6f} eV/Å")
        print(f"  Force   R²   = {metrics_f['r2']:.6f}")
        records.append({
            "quantity": "force", "rmse": metrics_f["rmse"],
            "mae": metrics_f["mae"], "r2": metrics_f["r2"],
            "n_frames": n_frames,
        })

    if pred_energy.size and ref_energy.size:
        metrics_e = _compute_metrics(pred_energy, ref_energy, "Energy")
        print(f"  Energy  RMSE = {metrics_e['rmse']:.6g} eV")
        print(f"  Energy  MAE  = {metrics_e['mae']:.6g} eV")
        print(f"  Energy  R²   = {metrics_e['r2']:.6g}")
        records.append({
            "quantity": "energy", "rmse": metrics_e["rmse"],
            "mae": metrics_e["mae"], "r2": metrics_e["r2"],
            "n_frames": n_e_frames,
        })
    print("=" * 60)

    # --- Save metrics CSV ---
    if records:
        import csv
        csv_path = os.path.join(output_dir, "test_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=records[0].keys())
            w.writeheader()
            w.writerows(records)
        print(f"\nSaved metrics: {csv_path}")

    # ------------------------------------------------------------------
    # Render plots
    # ------------------------------------------------------------------
    if pred_forces.size and ref_forces.size and "force" not in skip_plots:
        _force_parity_plot(pred_forces[:n_frames], ref_forces[:n_frames], output_dir)
        _force_error_hist(pred_forces[:n_frames], ref_forces[:n_frames], output_dir)
    else:
        print("[plot-test] skipping force plots (no data or --skip force)")

    if pred_energy.size and ref_energy.size and "energy" not in skip_plots:
        _energy_parity_plot(pred_energy[:n_e_frames], ref_energy[:n_e_frames],
                            output_dir)
        _energy_error_hist(pred_energy[:n_e_frames], ref_energy[:n_e_frames],
                           output_dir)
    else:
        print("[plot-test] skipping energy plots (no data or --skip energy)")

    print("\n[plot-test] done")
    return 0
