"""Path templating, file discovery, and trajectory filtering utilities.

Consolidates the small helpers that were duplicated across legacy scripts:
- ``{temp}`` substitution in directory / pattern strings
- glob-based trajectory file discovery
- particles/box_vectors CSV file pairing
- trajectory filtering (max-interval selection)
- ``ensure_dir`` helper
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    """``os.makedirs(path, exist_ok=True)`` — idempotent."""
    if path:
        os.makedirs(str(path), exist_ok=True)


def substitute_temp(template: str, temp: Optional[int]) -> str:
    """Replace ``{temp}`` with ``str(temp)``. If temp is None, strip the
    placeholder (and any surrounding path separators that become redundant)."""
    if template is None:
        return template
    if temp is None:
        # Remove optional separator before {temp} and the placeholder itself
        return re.sub(r"/?\{temp\}", "", template)
    return template.replace("{temp}", str(temp))


def join_path(*parts: str) -> str:
    """``os.path.join`` that ignores None parts."""
    return os.path.join(*[p for p in parts if p is not None])


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def glob_with_temp(directory: str, pattern: str,
                   temp: Optional[int] = None) -> List[str]:
    """Substitute ``{temp}`` into directory and pattern, then glob.

    Sorts the result for deterministic ordering. Used by cg-gen's
    ``scan_trajectory_files``.
    """
    directory = substitute_temp(directory, temp)
    pattern = substitute_temp(pattern, temp)
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    return files


def find_paired_csvs(data_dir: str,
                     temp: Optional[int] = None
                     ) -> List[Tuple[str, str]]:
    """Find ``(particles_csv, box_csv)`` pairs in a data directory.

    1:1 port of legacy ``find_csv_files`` from 03-trans_CGnpy_parall.py.
    """
    if temp is not None:
        data_dir = data_dir.replace("{temp}", str(temp))

    particles_files = sorted(glob.glob(os.path.join(data_dir, "*_particles.csv")))

    pairs: List[Tuple[str, str]] = []
    for particles_file in particles_files:
        basename = particles_file.replace("_particles.csv", "")
        box_file = f"{basename}_box_vectors.csv"
        if os.path.exists(box_file):
            pairs.append((particles_file, box_file))
    return pairs


def find_particle_files(base_dir: str) -> List[str]:
    """Recursively find all ``*_particles.csv`` files under ``base_dir``."""
    return sorted(glob.glob(os.path.join(base_dir, "**", "*_particles.csv"),
                            recursive=True))


# ---------------------------------------------------------------------------
# Trajectory filtering (ported from legacy 02-get_CGdata_parall.py)
# ---------------------------------------------------------------------------

def extract_timestep_from_filename(filename: str) -> Optional[int]:
    """Extract integer timestep from a trajectory filename.

    Looks for the last run of digits in the basename. Returns None on failure.
    """
    basename = os.path.basename(filename)
    match = re.search(r'(\d+)', basename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def select_trajectories_max_interval(files: List[str],
                                     max_trajectories: int) -> List[str]:
    """Select ``max_trajectories`` files approximately evenly across the
    sorted-by-timestep sequence. 1:1 port of legacy implementation."""
    if len(files) <= max_trajectories:
        return list(files)

    # Pair each file with its timestep, sort by timestep
    file_times = []
    for f in files:
        t = extract_timestep_from_filename(f)
        file_times.append((f, t if t is not None else 0))
    file_times.sort(key=lambda x: x[1])

    selected = []
    if max_trajectories <= 0:
        return selected

    # Even interval sampling
    n = len(file_times)
    if max_trajectories == 1:
        return [file_times[n // 2][0]]

    interval = (n - 1) / (max_trajectories - 1)
    for i in range(max_trajectories):
        idx = int(round(i * interval))
        selected.append(file_times[idx][0])
    return selected


def apply_trajectory_filter(files: List[str],
                            sim_name: str,
                            temp: Optional[int],
                            filter_config: Dict[str, Any]) -> List[str]:
    """Apply trajectory filtering based on config.

    Expected filter_config shape::

        {
          "enabled": true,
          "selection_method": "max_interval",
          "per_temp": {
            "default": {"max_trajectories": 10},
            "<sim_name>": {
              "temperatures": {"<temp>": 5, ...},
              "max_trajectories": 8
            }
          }
        }
    """
    if not filter_config or not filter_config.get("enabled", False):
        return list(files)
    if not files:
        return files

    method = filter_config.get("selection_method", "max_interval")
    if method != "max_interval":
        return list(files)

    per_temp = filter_config.get("per_temp", {}) or {}
    sim_cfg = per_temp.get(sim_name, {}) or {}
    default_cfg = per_temp.get("default", {}) or {}

    # Determine max_trajectories for this (sim, temp)
    max_traj = None
    if temp is not None:
        temp_cfg = sim_cfg.get("temperatures", {}) or {}
        max_traj = temp_cfg.get(str(temp))
    if max_traj is None:
        max_traj = sim_cfg.get("max_trajectories")
    if max_traj is None:
        max_traj = default_cfg.get("max_trajectories")

    if max_traj is None or max_traj <= 0:
        return list(files)

    return select_trajectories_max_interval(files, int(max_traj))
