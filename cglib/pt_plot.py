"""plot-pt domain logic: extract P/T thermo data and render overview plot.

Reads LAMMPS ``log.lammps`` thermo tables (via :func:`cglib.fparam.parse_lammps_thermo`)
and joins each dump frame's timestep to the measured ``Temp``/``Press``. Outputs
a tidy CSV plus a Nature-style P-vs-T scatter colored by ``sim_type``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .analyze_atomic import (
    AtomicStructureAnalyzer,
    _apply_lego_legend,
    _categorical_palette,
    _import_heavy_deps,
)
from .fparam import parse_lammps_thermo, query_thermo
from .lammps import LammpsDumpReader
from .paths import ensure_dir


# =============================================================================
# Table builder
# =============================================================================

def _build_pt_table(
    analyzer: AtomicStructureAnalyzer,
    log_dir: str,
    allowed_sims: Optional[List[str]],
    allowed_temps: Optional[List[int]],
    max_frames: Optional[int],
) -> pd.DataFrame:
    """Build a per-frame P/T table by joining dump timesteps to ``log.lammps``.

    For every AA dump discovered by :meth:`AtomicStructureAnalyzer.find_trajectory_files`
    the first frame's ``timestep`` is read; the parent ``sim`` directory tells
    us which ``log.lammps`` to consult. ``query_thermo`` then resolves the
    measured ``Temp`` / ``Press`` for that timestep (handling multi-block logs
    with ``reset_timestep`` between temperature sweeps).

    Args:
        analyzer: configured AtomicStructureAnalyzer (AA mode, base_dir set).
        log_dir: root that holds ``<sim>/log.lammps`` (typically ``paths.log_dir``).
        allowed_sims: optional ``--sim`` filter; ``None`` keeps all.
        allowed_temps: optional ``--temp`` filter on the path-derived nominal T.
        max_frames: optional cap on total rows (uniform downsample).

    Returns:
        DataFrame with columns ``structure_id, sim_type, temp_nominal,
        temp_measured, pressure, timestep, source_file``. ``temp_measured`` and
        ``pressure`` are ``NaN`` when ``log.lammps`` is missing or the
        timestep isn't found in any thermo block.
    """
    files = analyzer.find_trajectory_files()
    if not files:
        raise RuntimeError(
            f"No LAMMPS dump files under {analyzer.base_dir}. "
            "Pass --base-dir pointing at 01.aa (or a sim subdir)."
        )

    # Apply --sim / --temp filters using the same path parser the analyzer uses
    # so the user's CLI selection is honored regardless of layout.
    filtered: List[str] = []
    for f in files:
        sim_type, temp = analyzer._extract_sim_type_and_temp(f)
        if allowed_sims and sim_type not in allowed_sims:
            continue
        if allowed_temps and (temp is None or temp not in allowed_temps):
            continue
        filtered.append(f)
    files = filtered or files  # graceful fallback: empty filters keep all

    rows: List[Dict[str, Any]] = []
    thermo_cache: Dict[str, List[Dict[int, Dict[str, float]]]] = {}
    skip_log = 0
    no_match = 0

    for filepath in files:
        sim_type, temp_nominal = analyzer._extract_sim_type_and_temp(filepath)

        # Locate the sim's log.lammps by walking up until we find a sibling
        # of ``log_dir`` (i.e. parent of the dump's sim directory).
        # We assume paths like: <log_dir>/<sim>/traj/<dump> or <log_dir>/<sim>/<temp>/<dump>.
        sim_dir = _find_sim_dir(filepath, log_dir)
        log_path = os.path.join(sim_dir, 'log.lammps')

        timestep: Optional[int] = None
        try:
            reader = LammpsDumpReader(filepath)
            reader.parse_file()
            frame = reader.read_first_frame() if reader.frames else None
            if frame is not None:
                timestep = int(frame.get('timestep', 0))
        except Exception:
            timestep = None

        temp_measured = np.nan
        pressure = np.nan
        if not os.path.exists(log_path):
            skip_log += 1
        elif timestep is None:
            no_match += 1
        else:
            if sim_dir not in thermo_cache:
                thermo_cache[sim_dir] = parse_lammps_thermo(log_path)
            blocks = thermo_cache[sim_dir]
            row = query_thermo(blocks, temp_nominal, timestep)
            if row is None:
                no_match += 1
            else:
                temp_measured = float(row.get('Temp', np.nan))
                pressure = float(row.get('Press', np.nan))
                if np.isnan(temp_measured) or np.isnan(pressure):
                    no_match += 1

        rows.append({
            'structure_id': analyzer._build_structure_id(
                sim_type, temp_nominal, timestep if timestep is not None else -1),
            'sim_type': sim_type,
            'temp_nominal': temp_nominal if temp_nominal is not None else np.nan,
            'temp_measured': temp_measured,
            'pressure': pressure,
            'timestep': timestep if timestep is not None else -1,
            'source_file': filepath,
        })

    df = pd.DataFrame(rows)
    if max_frames is not None and len(df) > max_frames:
        # Uniform stride so the P-T cloud keeps its shape after culling.
        idx = np.linspace(0, len(df) - 1, num=max_frames, dtype=int)
        df = df.iloc[idx].reset_index(drop=True)

    if skip_log:
        print(f"[skip-log] {skip_log} frames had no log.lammps nearby")
    if no_match:
        print(f"[no-match] {no_match} frames' timestep not found in thermo blocks")
    return df


def _find_sim_dir(filepath: str, log_dir: str) -> str:
    """Locate the sim directory that owns ``filepath`` relative to ``log_dir``.

    We need the path that ``<sim>/log.lammps`` lives under. Strategy:
    walk up from the dump's parent until either we hit ``log_dir`` or until
    the parent's name matches a known sim-style directory (heuristic). Falls
    back to the dump's grandparent if no better candidate is found.
    """
    p = Path(filepath).resolve()
    log_root = Path(log_dir).resolve()
    # Walk up until our parent is log_dir (so we ARE the sim dir).
    for parent in p.parents:
        if parent.parent == log_root:
            return str(parent)
        # Stop one level above log_dir to avoid walking past the AA root.
        if parent == log_root:
            return str(parent)
    # Fallback: dump's grandparent usually corresponds to a sim dir for the
    # common <sim>/<temp>/<dump> and <sim>/traj/<dump> layouts.
    return str(p.parent.parent) if len(p.parents) >= 2 else str(p.parent)


# =============================================================================
# Plot
# =============================================================================

def _render_pt_scatter(df: pd.DataFrame, output_dir: str) -> None:
    """Render a single P-vs-T scatter colored by ``sim_type``.

    Loads matplotlib lazily so ``cgkit plot-pt`` doesn't pull heavy deps until
    a plot is actually requested.
    """
    _import_heavy_deps()
    import matplotlib.pyplot as plt  # noqa: F811  (safe; module-level plt is also set)

    out_dir = Path(output_dir)
    (out_dir / 'figures').mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    unique_sims = sorted(df['sim_type'].dropna().unique().tolist())
    colors = _categorical_palette(len(unique_sims))
    sim_color = {s: colors[i] for i, s in enumerate(unique_sims)}

    for sim in unique_sims:
        sub = df[df['sim_type'] == sim]
        ax.scatter(sub['temp_measured'], sub['pressure'],
                   c=[sim_color[sim]], label=sim, alpha=0.6, s=25,
                   edgecolors='none')

    ax.set_xlabel('Measured Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.set_title('P–T coverage of AA trajectories')
    _apply_lego_legend(ax, unique_sims, outside=len(unique_sims) > 6)
    fig.tight_layout()

    out_path = out_dir / 'figures' / 'pt_overview.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# Entry point
# =============================================================================

def run(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """cgkit plot-pt entry."""
    paths = config.get('paths', {})
    log_dir = (getattr(args, 'log_dir', None) or
               paths.get('log_dir', '/mnt/d/Workbench/CH_CG/01.aa'))

    # Output dir: try args, then plot_pt.output_dir, then paths.analysis_output_base_dir.
    plot_cfg = config.get('plot_pt', {}) or {}
    output_dir = (getattr(args, 'output_dir', None) or
                  plot_cfg.get('output_dir') or
                  paths.get('analysis_output_base_dir') or
                  '/mnt/d/Workbench/CH_CG/structure_analysis_results')
    output_dir = os.path.join(output_dir, 'pt_overview')
    ensure_dir(output_dir)

    base_dir = getattr(args, 'base_dir', None) or paths.get('aa_data_base_dir')
    max_frames = (getattr(args, 'max_frames', None) or
                  plot_cfg.get('max_frames') or
                  config.get('analysis_atomic', {}).get('max_frames') or
                  2000)

    analyzer = AtomicStructureAnalyzer(
        base_dir=base_dir, config=config, mode='aa', output_dir=output_dir,
    )

    allowed_sims = getattr(args, 'sim', None)
    allowed_temps = getattr(args, 'temp', None)

    print(f"[plot-pt] base_dir={analyzer.base_dir}")
    print(f"[plot-pt] log_dir={log_dir}")
    print(f"[plot-pt] output_dir={output_dir}")
    print(f"[plot-pt] max_frames={max_frames}")

    df = _build_pt_table(analyzer, log_dir, allowed_sims, allowed_temps, max_frames)

    csv_path = os.path.join(output_dir, 'pt_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({len(df)} rows)")
    if len(df) == 0:
        print("[plot-pt] empty table; skipping plot")
        return 1

    # Summary stats so the CLI is informative even without opening the PNG.
    valid = df.dropna(subset=['temp_measured', 'pressure'])
    if len(valid):
        print(f"  T_measured range : {valid['temp_measured'].min():.1f} – "
              f"{valid['temp_measured'].max():.1f} K")
        print(f"  Pressure range   : {valid['pressure'].min():.1f} – "
              f"{valid['pressure'].max():.1f} bar")
        print(f"  Coverage         : {len(valid)}/{len(df)} frames matched "
              f"({100 * len(valid) / len(df):.1f}%)")

    _render_pt_scatter(df, output_dir)
    return 0
