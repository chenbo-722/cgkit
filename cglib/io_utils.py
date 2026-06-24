"""Unified CSV / .raw I/O helpers for cglib.

Covers the read/write patterns shared by cg-gen (write particles / box CSV),
to-deepmd (read particles / box CSV, write DeepMD .raw / .npy), and the
analysis tools (read particles CSV).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .paths import ensure_dir


# ---------------------------------------------------------------------------
# CSV read / write (particles + box_vectors)
# ---------------------------------------------------------------------------

def read_particles_csv(path: str) -> pd.DataFrame:
    """Read a ``*_particles.csv`` file produced by cg-gen."""
    return pd.read_csv(path)


def read_box_vectors_csv(path: str) -> pd.DataFrame:
    """Read a ``*_box_vectors.csv`` file produced by cg-gen.

    Columns: ``timestep, x1, y1, z1, x2, y2, z2, x3, y3, z3``
    (or any column order; we return the raw DataFrame).
    """
    return pd.read_csv(path)


def write_particles_csv(df_particles: pd.DataFrame,
                        output_prefix: str,
                        filename_template: str = "{basename}_particles.csv"
                        ) -> str:
    """Write the per-frame particles DataFrame to disk. Returns the file path."""
    out_path = os.path.join(os.path.dirname(output_prefix),
                            filename_template.format(
                                basename=os.path.basename(output_prefix)))
    ensure_dir(os.path.dirname(out_path))
    df_particles.to_csv(out_path, index=False)
    return out_path


def write_box_vectors_csv(timestep: int,
                          box_vector_9: List[float],
                          output_prefix: str,
                          filename_template: str = "{basename}_box_vectors.csv"
                          ) -> str:
    """Append one row of (timestep, *box_vector_9) to the box vectors CSV.

    Mirrors legacy behaviour: each frame is appended in row form.
    """
    out_path = os.path.join(os.path.dirname(output_prefix),
                            filename_template.format(
                                basename=os.path.basename(output_prefix)))
    ensure_dir(os.path.dirname(out_path))
    row = [timestep] + list(box_vector_9)
    write_header = not os.path.exists(out_path)
    with open(out_path, "a") as f:
        if write_header:
            f.write("timestep,x1,y1,z1,x2,y2,z2,x3,y3,z3\n")
        f.write(",".join(str(v) for v in row) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# DeepMD .raw / .npy writers
# ---------------------------------------------------------------------------

def write_raw(path: str, array: np.ndarray) -> None:
    """Write a 1D or 2D ndarray to DeepMD .raw text format (space-separated)."""
    ensure_dir(os.path.dirname(path))
    if array.ndim == 1:
        lines = [f"{v}" for v in array.ravel()]
    else:
        lines = [" ".join(f"{v}" for v in row) for row in array]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_npy(path: str, array: np.ndarray) -> None:
    """Write an ndarray to .npy. Ensures the parent directory exists."""
    ensure_dir(os.path.dirname(path))
    np.save(path, array)


def write_type_raw(types_per_frame_first: List[int], out_dir: str) -> None:
    """Write the ``type.raw`` file (one frame's particle types, space-separated)."""
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "type.raw"), "w") as f:
        f.write(" ".join(str(int(t)) for t in types_per_frame_first) + "\n")


def write_type_map_raw(type_names: List[str], out_dir: str) -> None:
    """Write ``type_map.raw`` (space-separated CG type names, e.g. ``CG1 CG2``)."""
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "type_map.raw"), "w") as f:
        f.write(" ".join(type_names) + "\n")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def count_raw_lines(path: str) -> int:
    """Count non-empty lines in a text file. Used by fparam const to match
    the frame count of ``box.raw``."""
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip())
