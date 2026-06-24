"""fparam domain logic: generate fparam.raw / fparam.npy for DeepMD-kit.

Migrated from legacy ``generate_fparam.py``. Two modes (extract / const)
share ``--unit`` (K | eV). 1:1 functional port.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .paths import ensure_dir

# Boltzmann constant in eV/K (CODATA 2018)
K_B_EV_PER_K = 8.617333262e-5


# =============================================================================
# LAMMPS log parser (used by extract mode)
# =============================================================================

def parse_lammps_log(log_file: str) -> List[float]:
    """Parse LAMMPS log.lammps and return per-step temperatures (Kelvin)."""
    temps: List[float] = []
    in_thermo = False
    temp_idx: Optional[int] = None

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not in_thermo and 'Step' in line and 'Temp' in line:
                headers = line.split()
                if 'Temp' in headers:
                    temp_idx = headers.index('Temp')
                    in_thermo = True
                    continue
            if in_thermo:
                parts = line.split()
                if parts and parts[0].lstrip('-').isdigit():
                    try:
                        temps.append(float(parts[temp_idx]))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith('Loop time') or not parts:
                    in_thermo = False
    return temps


def to_output_unit(values_kelvin, unit: str) -> np.ndarray:
    arr = np.asarray(values_kelvin, dtype=np.float64)
    return arr * K_B_EV_PER_K if unit == 'eV' else arr


def write_fparam(values, raw_path: str, npy_path: str) -> None:
    ensure_dir(os.path.dirname(raw_path))
    ensure_dir(os.path.dirname(npy_path))
    with open(raw_path, 'w') as f:
        for v in values:
            f.write(f"{v:.10g}\n")
    np.save(npy_path, np.asarray(values, dtype=np.float64))


# =============================================================================
# extract mode
# =============================================================================

def run_extract(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """Extract per-frame T(t) from LAMMPS log files."""
    paths = config['paths']
    fparam_cfg = config.get('fparam', {})
    unit = args.unit if getattr(args, 'unit', None) else fparam_cfg.get('unit', 'eV')
    sim_names = (args.sim if getattr(args, 'sim', None)
                 else fparam_cfg.get('extract', {}).get('sim_names', ['3-upT', '4-dnT']))
    log_dir = paths.get('log_dir', '/mnt/d/Workbench/CH_CG/01.aa')
    output_dir = paths.get('deepmd_output_base_dir',
                           '/mnt/d/Workbench/CH_CG/03.cg_npy/training_data')

    print(f"Unit: {unit}  (kB = {K_B_EV_PER_K} eV/K)" if unit == 'eV'
          else "Unit: K  (Kelvin)")
    total = 0
    for sim_name in sim_names:
        log_file = os.path.join(log_dir, sim_name, 'log.lammps')
        raw_file = os.path.join(output_dir, sim_name, 'fparam.raw')
        npy_file = os.path.join(output_dir, sim_name, 'set.000', 'fparam.npy')

        if not os.path.exists(log_file):
            print(f"[skip] Log not found: {log_file}")
            continue

        print(f"[extract] {sim_name}  <- {log_file}")
        temps_K = parse_lammps_log(log_file)
        if not temps_K:
            print(f"  Warning: no temperature values found in {log_file}")
            continue

        values = to_output_unit(temps_K, unit)
        write_fparam(values, raw_file, npy_file)

        print(f"  frames : {len(values)}")
        print(f"  range  : {values.min():.6g} - {values.max():.6g} {unit}")
        print(f"  -> {raw_file}")
        print(f"  -> {npy_file}")
        total += len(values)

    print(f"\nTotal frames written (extract): {total}")
    return 0


# =============================================================================
# const mode
# =============================================================================

def run_const(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """Generate constant-T fparam matched to box.raw frame count."""
    paths = config['paths']
    fparam_cfg = config.get('fparam', {})
    unit = args.unit if getattr(args, 'unit', None) else fparam_cfg.get('unit', 'eV')
    sim_names = (args.sim if getattr(args, 'sim', None)
                 else fparam_cfg.get('const', {}).get('sim_names', ['1-npt', '2-nvt']))
    temperatures = (args.temp if getattr(args, 'temp', None)
                    else fparam_cfg.get('const', {}).get('temperatures',
                                                          [200, 300, 400, 500, 600]))
    base_dir = paths.get('deepmd_output_base_dir',
                         '/mnt/d/Workbench/CH_CG/03.cg_npy/training_data')

    print(f"Unit: {unit}  (kB = {K_B_EV_PER_K} eV/K)" if unit == 'eV'
          else "Unit: K  (Kelvin)")

    for sim_name in sim_names:
        print(f"[const] {sim_name}")
        for temperature in temperatures:
            temp_dir = os.path.join(base_dir, sim_name, str(temperature))
            box_file = os.path.join(temp_dir, 'box.raw')
            raw_file = os.path.join(temp_dir, 'fparam.raw')
            npy_file = os.path.join(temp_dir, 'set.000', 'fparam.npy')

            if not os.path.exists(temp_dir):
                print(f"  [skip] Directory not found: {temp_dir}")
                continue
            if not os.path.exists(box_file):
                print(f"  [skip] box.raw not found: {box_file}")
                continue

            with open(box_file, 'r') as f:
                n_frames = sum(1 for line in f if line.strip())

            value_K = float(temperature)
            value = to_output_unit([value_K], unit)[0]
            values = np.full(n_frames, value, dtype=np.float64)
            write_fparam(values, raw_file, npy_file)

            print(f"  T={value_K} K  ->  {value:.10g} {unit}  ({n_frames} frames)")
            print(f"    -> {raw_file}")
            print(f"    -> {npy_file}")
    return 0


# =============================================================================
# Entry dispatcher
# =============================================================================

def run(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """cgkit fparam entry. Dispatches by args.fparam_mode."""
    if getattr(args, 'fparam_mode', None) == 'const':
        return run_const(config, args)
    return run_extract(config, args)
