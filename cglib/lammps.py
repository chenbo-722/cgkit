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
