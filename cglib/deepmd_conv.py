"""to-deepmd domain logic: CG CSV -> DeepMD-kit .raw / .npy.

Migrated from legacy ``03-trans_CGnpy_parall.py``. Conversion algorithm
preserved 1:1; only infrastructure (config, parallel, paths) is delegated.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import parallel as _parallel
from .paths import ensure_dir, substitute_temp
from .paths import find_paired_csvs


# =============================================================================
# Per-file worker
# =============================================================================

def _process_one_case(task: Tuple[str, str],
                      conv_config: Dict[str, Any]) -> Dict[str, Any]:
    """Read one (particles, box) CSV pair and return DeepMD-shaped arrays.

    1:1 port of legacy 03-trans_CGnpy_parall._process_one_case.
    """
    particles_file, box_file = task

    df_box = pd.read_csv(box_file)
    box_info = df_box.iloc[:, :-1].values  # exclude timestep column
    if np.unique(box_info, axis=0).shape[0] > 1:
        print(f"Warning: Box varies across timesteps in {os.path.basename(box_file)}")
    box = box_info[-1, :].reshape(-1)

    df = pd.read_csv(particles_file)
    if 'timestep' in df.columns:
        timesteps = df['timestep'].unique()
    else:
        timesteps = [0]

    all_coords: List[np.ndarray] = []
    all_forces: List[np.ndarray] = []
    all_energies: List[np.ndarray] = []
    all_types: List[np.ndarray] = []

    for timestep in timesteps:
        if timestep != 0:
            df_ts = df[df['timestep'] == timestep]
        else:
            df_ts = df
        df_ts = df_ts.sort_values('id')

        coords = df_ts[['x', 'y', 'z']].values.reshape(-1)
        forces = df_ts[['fx', 'fy', 'fz']].values.reshape(-1)
        energies = df_ts['c_pe'].values.reshape(-1)

        use_type_column = conv_config.get("use_type_column", True)
        if use_type_column and 'type' in df_ts.columns:
            original_types = df_ts['type'].values.astype(int)
            unique_types = sorted(np.unique(original_types))
            type_mapping = {old: new for new, old in enumerate(unique_types)}
            types = np.array([type_mapping[t] for t in original_types], dtype=int)
        else:
            if 'n_atoms' in df_ts.columns:
                n_atoms = df_ts['n_atoms'].values
                unique_vals = np.unique(n_atoms)
                val_to_type = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                types = np.array([val_to_type.get(val, 0) for val in n_atoms], dtype=int)
            else:
                types = np.zeros(len(df_ts), dtype=int)

        all_coords.append(coords)
        all_forces.append(forces)
        all_energies.append(energies)
        all_types.append(types)

    return {
        'coords': np.array(all_coords),
        'forces': np.array(all_forces),
        'energies': np.array(all_energies),
        'box': box,
        'types': np.array(all_types),
        'n_frames': len(all_coords),
        'n_particles': len(df['id'].unique()) if 'id' in df.columns else len(df_ts),
    }


def _worker(task):
    """Picklable top-level wrapper for ProcessPoolExecutor."""
    pair, conv_config = task
    try:
        result = _process_one_case(pair, conv_config)
        return pair, True, '', result
    except Exception as exc:  # noqa: BLE001
        return pair, False, str(exc), {}


# =============================================================================
# Per-simulation orchestrator
# =============================================================================

def process_simulation(sim_config: Dict[str, Any],
                       paths_config: Dict[str, Any],
                       conv_config: Dict[str, Any],
                       proc_config: Dict[str, Any],
                       output_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single simulation. 1:1 port."""
    if not sim_config.get("enabled", True):
        return {'name': sim_config['name'], 'enabled': False,
                'total_files': 0, 'success': 0, 'failed': 0, 'errors': []}

    stats: Dict[str, Any] = {
        'name': sim_config['name'], 'enabled': True,
        'total_files': 0, 'success': 0, 'failed': 0,
        'total_frames': 0, 'total_particles': 0, 'errors': [],
    }

    temperatures = sim_config.get('temperatures') or [None]
    data_subdir = sim_config.get('data_subdir', '')
    out_subdir = sim_config.get('output_subdir', '')

    for temp in temperatures:
        temp_str = f"{temp}K" if temp is not None else "N/A"

        data_dir = os.path.join(paths_config['cg_data_base_dir'],
                                substitute_temp(data_subdir, temp))
        out_dir = os.path.join(paths_config['deepmd_output_base_dir'],
                               substitute_temp(out_subdir, temp))
        if not os.path.exists(data_dir):
            stats['errors'].append(f"Data directory does not exist: {data_dir}")
            continue

        file_pairs = find_paired_csvs(data_dir, temp)
        if not file_pairs:
            continue

        stats['total_files'] += len(file_pairs)
        if proc_config.get('create_output_dirs', True):
            ensure_dir(out_dir)

        tasks = [(pair, conv_config) for pair in file_pairs]
        results = _parallel.run_parallel(
            tasks, _worker,
            n_workers=proc_config.get('max_workers'),
            parallel=proc_config.get('parallel', True),
            desc=f"  {sim_config['name']} {temp_str}", unit="file",
        )

        all_coords, all_forces, all_energies = [], [], []
        all_box: List[np.ndarray] = []
        all_types: List[np.ndarray] = []

        for pair, ok, msg, result in results:
            if ok:
                all_coords.append(result['coords'])
                all_forces.append(result['forces'])
                all_energies.append(result['energies'])
                all_box.append(result['box'])
                all_types.append(result['types'])
                stats['success'] += 1
                stats['total_frames'] += result['n_frames']
                stats['total_particles'] += result['n_particles']
            else:
                stats['failed'] += 1
                stats['errors'].append(f"{os.path.basename(pair[0])}: {msg}")

        if not all_coords:
            continue

        all_coords_arr = np.concatenate(all_coords, axis=0)
        all_forces_arr = np.concatenate(all_forces, axis=0)
        all_energies_arr = np.concatenate(all_energies, axis=0)
        all_box_arr = np.array(all_box)
        all_types_arr = np.concatenate(all_types, axis=0)

        if len(all_coords_arr) == 0 or len(all_types_arr) == 0:
            stats['errors'].append("Empty data after concatenation")
            continue

        total_energy = (all_energies_arr if all_energies_arr.ndim == 1
                        else np.sum(all_energies_arr, axis=1))

        _write_deepmd_output(out_dir, all_box_arr, all_coords_arr,
                             all_forces_arr, all_energies_arr, total_energy,
                             all_types_arr, conv_config, output_config)

    return stats


def _write_deepmd_output(out_dir: str,
                         all_box: np.ndarray,
                         all_coords: np.ndarray,
                         all_forces: np.ndarray,
                         all_energies: np.ndarray,
                         total_energy: np.ndarray,
                         all_types: np.ndarray,
                         conv_config: Dict[str, Any],
                         output_config: Dict[str, Any]) -> None:
    """Write the DeepMD .raw + .npy files. 1:1 port."""
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "set.000"))

    if output_config.get("save_raw_files", True):
        np.savetxt(os.path.join(out_dir, "box.raw"),       all_box,       fmt='%.8f', delimiter=' ')
        np.savetxt(os.path.join(out_dir, "coord.raw"),     all_coords,    fmt='%.8f', delimiter=' ')
        np.savetxt(os.path.join(out_dir, "force.raw"),     all_forces,    fmt='%.8f', delimiter=' ')
        np.savetxt(os.path.join(out_dir, "atom_ener.raw"), all_energies,  fmt='%.8f', delimiter=' ')
        np.savetxt(os.path.join(out_dir, "energy.raw"),    total_energy,  fmt='%.8f', delimiter=' ')

        try:
            if all_types.ndim > 1:
                first_frame_types = all_types[0, :]
            elif all_types.ndim == 1:
                first_frame_types = all_types
            else:
                first_frame_types = all_types.flatten()[:all_coords.shape[1] // 3]

            with open(os.path.join(out_dir, "type.raw"), 'w') as f:
                f.write(' '.join(map(str, first_frame_types.tolist())) + "\n")

            unique_types = np.unique(first_frame_types)
            type_map_str = ' '.join([f"CG{int(t)}" for t in sorted(unique_types)])
            with open(os.path.join(out_dir, "type_map.raw"), 'w') as f:
                f.write(type_map_str + "\n")
        except Exception as exc:  # noqa: BLE001
            n_particles = (all_coords.shape[1] // 3 if all_coords.ndim > 1
                           else len(all_types))
            default_types = np.ones(n_particles, dtype=int)
            with open(os.path.join(out_dir, "type.raw"), 'w') as f:
                f.write(' '.join(map(str, default_types.tolist())) + "\n")
            with open(os.path.join(out_dir, "type_map.raw"), 'w') as f:
                f.write("CG1\n")

    if output_config.get("save_npy_files", True):
        groups = max(1, conv_config.get("num_groups", 1))
        splits_coords = np.array_split(all_coords, groups, axis=0)
        splits_forces = np.array_split(all_forces, groups, axis=0)
        splits_ener   = np.array_split(all_energies, groups, axis=0)
        splits_box    = np.array_split(all_box, groups, axis=0)
        splits_total  = np.array_split(total_energy, groups, axis=0)
        for g in range(groups):
            set_dir = os.path.join(out_dir, f"set.{g:03d}")
            ensure_dir(set_dir)
            np.save(os.path.join(set_dir, "box.npy"),       splits_box[g])
            np.save(os.path.join(set_dir, "coord.npy"),     splits_coords[g])
            np.save(os.path.join(set_dir, "force.npy"),     splits_forces[g])
            np.save(os.path.join(set_dir, "atom_ener.npy"), splits_ener[g])
            np.save(os.path.join(set_dir, "energy.npy"),    splits_total[g])


# =============================================================================
# Entry point
# =============================================================================

def run(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """cgkit to-deepmd entry point."""
    paths_config = config['paths']
    conv_config = config.get('deepmd', config.get('conversion', {}))
    proc_config = config.get('processing', {})
    output_config = config.get('output', {})

    print(f"\n{'=' * 60}")
    print("CSV TO DEEPMD-KIT FORMAT CONVERSION")
    print(f"{'=' * 60}")
    print(f"CG data directory: {paths_config['cg_data_base_dir']}")
    print(f"Output directory:  {paths_config['deepmd_output_base_dir']}")
    print(f"Use type column:   {conv_config.get('use_type_column', True)}")
    print(f"Number of groups:  {conv_config.get('num_groups', 1)}")
    print(f"{'=' * 60}\n")

    all_stats: List[Dict[str, Any]] = []
    for sim_config in config.get('simulations', []):
        if not sim_config.get("enabled", True):
            continue
        stats = process_simulation(sim_config, paths_config, conv_config,
                                   proc_config, output_config)
        all_stats.append(stats)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    total_files = total_success = total_failed = 0
    total_frames = total_particles = 0
    for stats in all_stats:
        if not stats['enabled']:
            continue
        total_files += stats['total_files']
        total_success += stats['success']
        total_failed += stats['failed']
        total_frames += stats['total_frames']
        total_particles += stats['total_particles']
        if stats['total_files'] > 0:
            print(f"\n{stats['name']}:")
            print(f"  Files: {stats['success']}/{stats['total_files']} processed")
            if stats['failed'] > 0:
                print(f"  Failed: {stats['failed']}")
            if stats['total_frames'] > 0:
                print(f"  Frames: {stats['total_frames']:,}")
            if stats['total_particles'] > 0:
                print(f"  Particles: {stats['total_particles']:,}")

    print(f"\n{'=' * 60}")
    print("TOTAL:")
    print(f"  Files: {total_success}/{total_files} processed")
    if total_failed > 0:
        print(f"  Failed: {total_failed}")
    if total_frames > 0:
        print(f"  Frames: {total_frames:,}")
    if total_particles > 0:
        print(f"  Particles: {total_particles:,}")
    print(f"{'=' * 60}\n")

    if any(s['errors'] for s in all_stats):
        print("Errors:")
        for stats in all_stats:
            for err in stats['errors']:
                print(f"  [{stats['name']}] {err}")
        print()

    return 0 if total_failed == 0 else 1
