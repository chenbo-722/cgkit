"""Unified LAMMPS dump file parser.

Replaces three near-duplicate implementations from the legacy scripts:
- ``LAMMPSTrajectoryParser``  (legacy 02-get_CGdata_parall.py)
- ``LAMMPSTrajectoryReader``  (legacy 0x-analyze_atomic_structure.py, AA mode)
- ``CGTrajectoryReader``      (legacy 0x-analyze_atomic_structure.py, CG mode)

A single ``LammpsDumpReader`` parses the file into ``LammpsFrame`` objects.
Three adapter methods provide the exact return shapes the legacy consumers
expect, so domain logic (cg-gen coarse-graining, atomic-structure analysis)
does not change:

- ``get_dataframe(idx)``    -> ``pandas.DataFrame``  (cg-gen compatibility)
- ``read_first_frame()``    -> ``dict`` with ndarray fields (analyze-atomic AA)
- ``read_all_frames()``     -> ``list[dict]``         (analyze-atomic CG)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class LammpsFrame:
    """One parsed frame of a LAMMPS dump file."""
    timestep: int
    natoms: int
    box_bounds: List[List[float]]          # [[xlo,xhi,...], ...]  (3x2 or 3x3)
    columns: List[str]                     # column names from "ITEM: ATOMS" line
    atoms: List[Dict[str, float]] = field(default_factory=list)
    # ``atoms`` mirrors the legacy representation: list of dicts keyed by column name.


class LammpsDumpReader:
    """Parse a LAMMPS dump (.lammpstrj) file once; expose legacy-shaped views."""

    def __init__(self, filepath: str | Path):
        self.filepath = str(filepath)
        self.frames: List[LammpsFrame] = []
        # Parallel arrays kept for backward compatibility with cg-gen's parser API
        self.timesteps: List[int] = []
        self.natoms: List[int] = []
        self.box_bounds: List[List[List[float]]] = []
        self.atoms_data: List[List[Dict[str, float]]] = []

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def parse_file(self, verbose: bool = False) -> bool:
        """Parse the whole file. Returns True on success."""
        if verbose:
            print(f"Parsing file: {self.filepath}")

        try:
            with open(self.filepath, "r") as f:
                lines = f.readlines()
        except OSError as exc:
            if verbose:
                print(f"Error opening file: {exc}")
            return False

        i = 0
        n_lines = len(lines)
        # Per-frame state
        timestep: Optional[int] = None
        natoms: Optional[int] = None
        box: Optional[List[List[float]]] = None

        while i < n_lines:
            line = lines[i].strip()

            if line == "ITEM: TIMESTEP":
                i += 1
                timestep = int(lines[i].strip())
                i += 1
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                i += 1
                natoms = int(lines[i].strip())
                i += 1
            elif line.startswith("ITEM: BOX BOUNDS"):
                i += 1
                box = []
                for _ in range(3):
                    parts = lines[i].strip().split()
                    # 2 floats (lo hi) or 3 (lo hi tilt); we keep first 2.
                    box.append([float(parts[0]), float(parts[1])])
                    i += 1
            elif line.startswith("ITEM: ATOMS"):
                columns = line.split()[2:]
                i += 1
                atoms: List[Dict[str, float]] = []
                for _ in range(natoms or 0):
                    if i >= n_lines:
                        break
                    parts = lines[i].strip().split()
                    if parts:
                        atoms.append({col: float(val)
                                      for col, val in zip(columns, parts)})
                    i += 1

                if timestep is not None and natoms is not None and box is not None:
                    frame = LammpsFrame(timestep=timestep, natoms=natoms,
                                        box_bounds=box, columns=columns,
                                        atoms=atoms)
                    self.frames.append(frame)
                    self.timesteps.append(timestep)
                    self.natoms.append(natoms)
                    self.box_bounds.append(box)
                    self.atoms_data.append(atoms)
                # Reset per-frame state for next frame
                timestep = None
                natoms = None
                box = None
            else:
                i += 1

        if verbose:
            print(f"Successfully parsed {len(self.frames)} timesteps")
        return True

    # ------------------------------------------------------------------
    # Legacy API: cg-gen compatibility (LAMMPSTrajectoryParser.get_dataframe)
    # ------------------------------------------------------------------
    def get_dataframe(self, timestep_index: int = -1) -> Optional[pd.DataFrame]:
        """Return the atoms of one frame as a DataFrame (legacy cg-gen shape)."""
        if not self.atoms_data:
            return None
        if timestep_index < 0:
            timestep_index = len(self.timesteps) + timestep_index
        if timestep_index >= len(self.timesteps) or timestep_index < 0:
            return None
        return pd.DataFrame(self.atoms_data[timestep_index])

    # ------------------------------------------------------------------
    # Legacy API: analyze-atomic AA mode (LAMMPSTrajectoryReader.read_frame)
    # ------------------------------------------------------------------
    def read_first_frame(self) -> Optional[Dict[str, Any]]:
        """Return the first frame as a dict with ndarray fields.

        Output shape (matches legacy LAMMPSTrajectoryReader.read_frame)::

            {
              'timestep': int,
              'positions': (N, 3) ndarray,
              'types':     (N,)   int ndarray,
              'ids':       (N,)   int ndarray,
              'box':       (3, 2) float ndarray,
              'energies':  (N,)   float ndarray or None,
              'forces':    (N, 3) float ndarray or None,
            }
        """
        if not self.frames:
            return None
        return self._frame_to_atomic_dict(self.frames[0])

    # ------------------------------------------------------------------
    # Legacy API: analyze-atomic CG mode (CGTrajectoryReader.read_all_frames)
    # ------------------------------------------------------------------
    def read_all_frames(self) -> List[Dict[str, Any]]:
        """Return a list of atomic-dict frames (legacy CGTrajectoryReader shape)."""
        return [self._frame_to_atomic_dict(frame) for frame in self.frames]

    # ------------------------------------------------------------------
    # Internal: column-dict frame -> ndarray-dict frame
    # ------------------------------------------------------------------
    @staticmethod
    def _frame_to_atomic_dict(frame: LammpsFrame) -> Dict[str, Any]:
        """Adapt LammpsFrame to the dict shape expected by the atomic analyzer."""
        atoms = frame.atoms
        n = len(atoms)
        if n == 0:
            return {
                'timestep': frame.timestep, 'positions': np.zeros((0, 3)),
                'types': np.zeros((0,), dtype=int),
                'ids': np.zeros((0,), dtype=int),
                'box': np.array(frame.box_bounds, dtype=float),
                'energies': None, 'forces': None,
            }

        # Build a column-major dict so we can pull arrays by name.
        col_set = set(frame.columns)
        get_col = lambda name: np.array([a.get(name, np.nan) for a in atoms])

        ids = get_col('id').astype(int)
        types = get_col('type').astype(int)

        # Prefer unwrapped coords; fall back to wrapped.
        if {'xu', 'yu', 'zu'}.issubset(col_set):
            positions = np.column_stack([get_col('xu'), get_col('yu'), get_col('zu')])
        elif {'x', 'y', 'z'}.issubset(col_set):
            positions = np.column_stack([get_col('x'), get_col('y'), get_col('z')])
        else:
            positions = np.zeros((n, 3))

        energies = get_col('c_pe') if 'c_pe' in col_set else None
        forces = None
        if {'fx', 'fy', 'fz'}.issubset(col_set):
            forces = np.column_stack([get_col('fx'), get_col('fy'), get_col('fz')])

        return {
            'timestep': frame.timestep,
            'positions': positions.astype(float),
            'types': types,
            'ids': ids,
            'box': np.array(frame.box_bounds, dtype=float),
            'energies': energies,
            'forces': forces,
        }


# =============================================================================
# Frame writer (symmetric with parse_file)
# =============================================================================

def write_lammps_frame(frame: LammpsFrame, output_path: str) -> None:
    """Write a single ``LammpsFrame`` to a LAMMPS dump file.

    The output format is the mirror image of what :meth:`parse_file` consumes::

        ITEM: TIMESTEP
        <timestep>
        ITEM: NUMBER OF ATOMS
        <natoms>
        ITEM: BOX BOUNDS pp pp pp
        <xlo> <xhi> <xy_tilt>
        <ylo> <yhi> <xz_tilt>
        <zlo> <zhi> <yz_tilt>
        ITEM: ATOMS <columns...>
        <one row per atom>

    ``frame.box_bounds`` is stored as ``[[xlo,xhi],[ylo,yhi],[zlo,zhi]]``
    (2 entries per axis, matching the parser). We append a ``0.0`` tilt column
    so each BOX BOUNDS line has 3 floats — the standard orthogonal-box form.
    """
    from .paths import ensure_dir

    ensure_dir(os.path.dirname(os.path.abspath(output_path)) or '.')

    box = frame.box_bounds
    # Each axis: [lo, hi] (+ optional tilt we don't store); pad tilt with 0.0.
    # repr() = shortest string that round-trips to the same double.
    box_lines = []
    for axis in range(3):
        lo = float(box[axis][0])
        hi = float(box[axis][1])
        tilt = float(box[axis][2]) if len(box[axis]) > 2 else 0.0
        box_lines.append(f"{lo!r} {hi!r} {tilt!r}")

    columns = frame.columns

    with open(output_path, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{frame.timestep}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{frame.natoms}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for line in box_lines:
            f.write(line + "\n")
        f.write("ITEM: ATOMS " + " ".join(columns) + "\n")
        for atom in frame.atoms:
            row = [_format_atom_value(atom.get(col)) for col in columns]
            f.write(" ".join(row) + "\n")


def _format_atom_value(value) -> str:
    """Render an atom-field value back to dump text.

    Integer-valued fields (id, type) print without trailing ``.0``; all other
    floats use ``repr()`` which gives the shortest string that parses back to
    the exact same double (Python 3 float repr contract) — i.e. lossless
    roundtrip. ``None``/missing values become ``0`` to keep the column count
    consistent.
    """
    if value is None:
        return "0"
    iv = int(value)
    if iv == value:
        return str(iv)
    return repr(float(value))
