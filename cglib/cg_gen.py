"""cg-gen domain logic: LAMMPS atomic trajectory -> coarse-grained CSV.

Migrated from legacy ``02-get_CGdata_parall.py``. All coarse-graining
algorithm logic is preserved 1:1; only the surrounding infrastructure
(config loading, parallel execution, LAMMPS parsing, file discovery,
trajectory filtering) is delegated to other cglib modules.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import paths as _paths
from . import parallel as _parallel
from .lammps import LammpsDumpReader
from .paths import (
    apply_trajectory_filter,
    ensure_dir,
    glob_with_temp,
)


# =============================================================================
# Distance calculation
# =============================================================================

def calculate_distance(pos1: Dict[str, float], pos2: Dict[str, float],
                       box_bounds: List[List[float]],
                       use_unwrapped: bool = True) -> float:
    """Squared atom-atom distance with minimum-image PBC.

    Returns *squared* distance to match the legacy behaviour (callers compare
    without squaring). 1:1 port from legacy 02-get_CGdata_parall.py.
    """
    if use_unwrapped:
        x1, y1, z1 = (pos1.get('xu', pos1.get('x', 0)),
                      pos1.get('yu', pos1.get('y', 0)),
                      pos1.get('zu', pos1.get('z', 0)))
        x2, y2, z2 = (pos2.get('xu', pos2.get('x', 0)),
                      pos2.get('yu', pos2.get('y', 0)),
                      pos2.get('zu', pos2.get('z', 0)))
    else:
        x1, y1, z1 = pos1.get('x', 0), pos1.get('y', 0), pos1.get('z', 0)
        x2, y2, z2 = pos2.get('x', 0), pos2.get('y', 0), pos2.get('z', 0)

    lx = box_bounds[0][1] - box_bounds[0][0]
    ly = box_bounds[1][1] - box_bounds[1][0]
    lz = box_bounds[2][1] - box_bounds[2][0]

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dx = dx - round(dx / lx) * lx
    dy = dy - round(dy / ly) * ly
    dz = dz - round(dz / lz) * lz
    return dx * dx + dy * dy + dz * dz


# =============================================================================
# Coarse-graining
# =============================================================================

def coarse_grain_trajectory(parser: LammpsDumpReader,
                            timestep_index: int = -1,
                            cg_config: Optional[Dict[str, Any]] = None
                            ) -> Tuple[Optional[List[Dict]], Optional[List[float]]]:
    """灵活模式粗粒化 — 1:1 port from legacy 02-."""
    if cg_config is None:
        cg_config = {
            "method": "flexible_pattern",
            "patterns": [[1, 2, 2], [1, 2, 2, 2]],
            "center_atom_type": 1,
            "position_source": "unwrapped",
            "average_forces": True,
            "average_potential_energy": True,
        }

    if not parser.atoms_data:
        return None, None

    if timestep_index < 0:
        timestep_index = len(parser.timesteps) + timestep_index
    if timestep_index >= len(parser.timesteps):
        return None, None

    atoms = parser.atoms_data[timestep_index]
    box_bounds = parser.box_bounds[timestep_index]
    box_vector = [
        box_bounds[0][1] - box_bounds[0][0], 0, 0,
        0, box_bounds[1][1] - box_bounds[1][0], 0,
        0, 0, box_bounds[2][1] - box_bounds[2][0],
    ]

    df = pd.DataFrame(atoms)
    patterns = cg_config.get("patterns", [[1, 2, 2]])
    center_type = cg_config.get("center_atom_type", 1)
    position_source = cg_config.get("position_source", "unwrapped")
    use_unwrapped = (position_source == "unwrapped")
    has_unwrapped = all(col in df.columns for col in ['xu', 'yu', 'zu'])
    cg_assignments = cg_config.get("cg_assignments", [])

    if use_unwrapped and not has_unwrapped:
        use_unwrapped = False

    coarse_particles: List[Dict[str, Any]] = []
    cg_id = 1
    used_atom_indices = set()

    def get_position(row):
        pos = {}
        if 'xu' in df.columns and 'yu' in df.columns and 'zu' in df.columns:
            pos['xu'] = row['xu']; pos['yu'] = row['yu']; pos['zu'] = row['zu']
            pos['x'] = row.get('x', row['xu'])
            pos['y'] = row.get('y', row['yu'])
            pos['z'] = row.get('z', row['zu'])
        elif 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            pos['x'] = row['x']; pos['y'] = row['y']; pos['z'] = row['z']
            pos['xu'] = row['x']; pos['yu'] = row['y']; pos['zu'] = row['z']
        return pos

    def create_cg_particle(atom_rows, cg_type, center_idx=0):
        nonlocal cg_id
        force_energy_source = cg_config.get("force_energy_source", "average")
        center_row = atom_rows.iloc[center_idx]
        pos = get_position(center_row)
        if use_unwrapped:
            pos_x, pos_y, pos_z = pos['xu'], pos['yu'], pos['zu']
        else:
            pos_x, pos_y, pos_z = pos['x'], pos['y'], pos['z']

        if force_energy_source == "center_only":
            avg_fx = center_row['fx']
            avg_fy = center_row['fy']
            avg_fz = center_row['fz']
            avg_pe = center_row['c_pe'] if 'c_pe' in df.columns else None
        else:
            if cg_config.get("average_forces", True):
                avg_fx = atom_rows['fx'].mean()
                avg_fy = atom_rows['fy'].mean()
                avg_fz = atom_rows['fz'].mean()
            else:
                avg_fx = avg_fy = avg_fz = 0
            avg_pe = None
            if cg_config.get("average_potential_energy", True) and 'c_pe' in df.columns:
                avg_pe = atom_rows['c_pe'].mean()

        return {
            'id': cg_id,
            'type': cg_type,
            'x': pos_x, 'y': pos_y, 'z': pos_z,
            'fx': avg_fx, 'fy': avg_fy, 'fz': avg_fz,
            'c_pe': avg_pe,
            'n_atoms': len(atom_rows),
            'atom_indices': list(atom_rows.index),
        }

    # Step 1: manual assignments
    for assignment in cg_assignments:
        atom_ids = assignment.get("atom_ids", [])
        assigned_cg_type = assignment.get("cg_type", 1)

        target_atoms = None
        if 'id' in df.columns:
            target_atoms = df[df['id'].isin(atom_ids)]
        else:
            target_indices = [aid - 1 for aid in atom_ids if aid - 1 < len(df)]
            target_atoms = df.loc[target_indices]

        if target_atoms is None or len(target_atoms) == 0:
            continue
        if any(idx in used_atom_indices for idx in target_atoms.index):
            continue

        particle = create_cg_particle(target_atoms, assigned_cg_type)
        particle['manual_assignment'] = True
        particle['assigned_atom_ids'] = atom_ids
        coarse_particles.append(particle)
        used_atom_indices.update(target_atoms.index)
        cg_id += 1

    # Step 2: pattern-based for remaining atoms
    pattern_to_cg_type: Dict[Tuple[int, ...], int] = {}
    cg_type_counter = 1
    for pattern in patterns:
        pattern_tuple = tuple(pattern)
        if pattern_tuple not in pattern_to_cg_type:
            pattern_to_cg_type[pattern_tuple] = cg_type_counter
            cg_type_counter += 1

    available_df = df[~df.index.isin(used_atom_indices)]
    center_atoms = available_df[available_df['type'] == center_type]

    if len(center_atoms) == 0 and len(coarse_particles) == 0:
        return [], box_vector

    for _, center_row in center_atoms.iterrows():
        if center_row.name in used_atom_indices:
            continue
        center_pos = get_position(center_row)
        if not center_pos:
            continue

        best_pattern = None
        best_match_atoms = None
        best_score = -1

        for pattern in patterns:
            pattern_length = len(pattern)
            n_neighbor_type2 = pattern.count(2)
            available_type2 = available_df[
                (available_df['type'] == 2)
                & (~available_df.index.isin(used_atom_indices))
            ]
            if len(available_type2) < n_neighbor_type2:
                continue

            distances = []
            for idx, type2_row in available_type2.iterrows():
                type2_pos = get_position(type2_row)
                if not type2_pos:
                    continue
                dist_sq = calculate_distance(center_pos, type2_pos, box_bounds, use_unwrapped)
                distances.append((dist_sq, idx))

            if len(distances) < n_neighbor_type2:
                continue

            distances.sort(key=lambda x: x[0])
            selected_type2_indices = [idx for _, idx in distances[:n_neighbor_type2]]
            selected_atoms = [center_row.name] + selected_type2_indices
            selected_atom_rows = df.loc[selected_atoms]
            score = len(selected_atoms) - pattern_length * 0.01
            if score > best_score:
                best_score = score
                best_pattern = pattern
                best_match_atoms = selected_atom_rows

        if best_match_atoms is None:
            continue

        pattern_tuple = tuple(best_pattern)
        cg_type = pattern_to_cg_type.get(pattern_tuple, 1)
        particle = create_cg_particle(best_match_atoms, cg_type)
        particle['pattern'] = best_pattern
        coarse_particles.append(particle)
        used_atom_indices.update(best_match_atoms.index)
        cg_id += 1

    return coarse_particles, box_vector


def process_all_timesteps(parser: LammpsDumpReader,
                          cg_config: Optional[Dict[str, Any]] = None
                          ) -> Tuple[List[Dict], np.ndarray]:
    """处理所有时间步 — 1:1 port."""
    all_coarse_data: List[Dict[str, Any]] = []
    all_box_vectors: List[List[float]] = []
    for timestep_idx in range(len(parser.timesteps)):
        coarse_particles, box_vector = coarse_grain_trajectory(parser, timestep_idx, cg_config)
        if coarse_particles is not None:
            for particle in coarse_particles:
                particle['timestep'] = parser.timesteps[timestep_idx]
            all_coarse_data.extend(coarse_particles)
            all_box_vectors.append(box_vector)
    return all_coarse_data, np.array(all_box_vectors)


# =============================================================================
# Data export
# =============================================================================

def export_coarse_grained_data(parser: LammpsDumpReader,
                               coarse_df: pd.DataFrame,
                               box_vectors: np.ndarray,
                               output_prefix: str,
                               output_config: Optional[Dict[str, Any]] = None,
                               verbose: bool = False) -> None:
    """Export particles + box_vectors CSV — 1:1 port."""
    if output_config is None:
        output_config = {"save_particles": True, "save_box_vectors": True}

    basename = os.path.basename(output_prefix)
    if output_config.get("save_particles", True):
        tpl = output_config.get("particles_filename", "{basename}_particles.csv")
        fname = tpl.format(basename=basename)
        path = os.path.join(os.path.dirname(output_prefix), fname)
        ensure_dir(os.path.dirname(path))
        coarse_df.to_csv(path, index=False)
        if verbose:
            print(f"  Particles data: {fname}")

    if output_config.get("save_box_vectors", True):
        tpl = output_config.get("box_vectors_filename", "{basename}_box_vectors.csv")
        fname = tpl.format(basename=basename)
        path = os.path.join(os.path.dirname(output_prefix), fname)
        ensure_dir(os.path.dirname(path))
        box_df = pd.DataFrame(box_vectors, columns=[
            'xlo', 'xhi', 'xy', 'ylo', 'yhi', 'xz', 'zlo', 'zhi', 'yz'
        ])
        box_df['timestep'] = parser.timesteps[:len(box_vectors)]
        box_df.to_csv(path, index=False)
        if verbose:
            print(f"  Box vectors: {fname}")


def export_cg_trajectory(parser: LammpsDumpReader,
                        all_coarse_data: List[Dict[str, Any]],
                        all_box_vectors: np.ndarray,
                        output_path: str,
                        cg_config: Optional[Dict[str, Any]] = None) -> None:
    """Export CG trajectory in LAMMPS dump format — 1:1 port."""
    if cg_config is None:
        cg_config = {}
    timesteps = sorted(set(p['timestep'] for p in all_coarse_data))
    timestep_to_particles = {
        ts: [p for p in all_coarse_data if p['timestep'] == ts] for ts in timesteps
    }
    use_unwrapped = (cg_config.get("position_source", "unwrapped") == "unwrapped")

    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        for ts_idx, timestep in enumerate(timesteps):
            particles = timestep_to_particles[timestep]
            box_vector = all_box_vectors[ts_idx]
            xlo, xhi = 0, box_vector[0]
            ylo, yhi = 0, box_vector[4]
            zlo, zhi = 0, box_vector[8]
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(particles)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{xlo:.6f} {xhi:.6f} 0.0\n")
            f.write(f"{ylo:.6f} {yhi:.6f} 0.0\n")
            f.write(f"{zlo:.6f} {zhi:.6f} 0.0\n")
            if use_unwrapped:
                f.write("ITEM: ATOMS id type xu yu zu fx fy fz\n")
            else:
                f.write("ITEM: ATOMS id type x y z fx fy fz\n")
            for p in sorted(particles, key=lambda x: x['id']):
                cg_type = p.get('type', 1)
                f.write(f"{p['id']} {cg_type} ")
                f.write(f"{p['x']:.6f} {p['y']:.6f} {p['z']:.6f} ")
                f.write(f"{p['fx']:.6f} {p['fy']:.6f} {p['fz']:.6f}\n")


# =============================================================================
# Per-file worker
# =============================================================================

def traj2CG(filename: str, out_dir: str,
            cg_config: Dict[str, Any],
            output_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single trajectory file — 1:1 port."""
    parser = LammpsDumpReader(filename)
    if not parser.parse_file(verbose=False):
        raise Exception("File parsing failed!")
    all_coarse_data, all_box_vectors = process_all_timesteps(parser, cg_config)
    coarse_df = pd.DataFrame(all_coarse_data)

    output_prefix = os.path.join(out_dir, os.path.basename(filename))
    export_coarse_grained_data(parser, coarse_df, all_box_vectors,
                               output_prefix, output_config, verbose=False)

    if cg_config.get("export_cg_trajectory", False):
        tpl = cg_config.get("cg_trajectory_filename", "{basename}_cg.lammpstrj")
        cg_traj_name = tpl.format(basename=os.path.basename(filename))
        cg_traj_path = os.path.join(out_dir, cg_traj_name)
        export_cg_trajectory(parser, all_coarse_data, all_box_vectors,
                             cg_traj_path, cg_config)

    return {
        'timesteps': len(all_box_vectors),
        'particles': len(coarse_df),
    }


def _process_one_case(task: Tuple[str, str, Dict[str, Any], Dict[str, Any]]
                      ) -> Tuple[str, bool, str, Dict[str, Any]]:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    in_file, out_dir, cg_config, output_config = task
    try:
        stats = traj2CG(in_file, out_dir, cg_config, output_config)
        return in_file, True, '', stats
    except Exception as exc:  # noqa: BLE001
        return in_file, False, str(exc), {}


# =============================================================================
# Per-simulation orchestrator
# =============================================================================

def process_simulation(sim_config: Dict[str, Any],
                       paths_config: Dict[str, Any],
                       cg_config: Dict[str, Any],
                       output_config: Dict[str, Any],
                       processing_config: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个模拟类型的所有轨迹文件 — 1:1 port."""
    stats: Dict[str, Any] = {
        'name': sim_config['name'],
        'enabled': sim_config.get("enabled", True),
        'total_files': 0, 'filtered_files': 0,
        'success': 0, 'failed': 0,
        'total_timesteps': 0, 'total_particles': 0,
        'errors': [],
    }
    if not stats['enabled']:
        return stats

    traj_dir = os.path.join(paths_config['base_dir'], sim_config['trajectory_dir'])
    if not os.path.exists(traj_dir):
        stats['errors'].append(f"Trajectory directory does not exist: {traj_dir}")
        return stats

    filter_config = processing_config.get('trajectory_filter', {}) or {}
    temperatures = sim_config.get('temperatures') or [None]

    for temp in temperatures:
        temp_str = f"{temp}K" if temp is not None else "N/A"
        files = glob_with_temp(traj_dir, sim_config['file_pattern'], temp)
        if not files:
            continue

        original_count = len(files)
        files = apply_trajectory_filter(files, sim_config['name'], temp, filter_config)
        filtered_count = len(files)
        stats['total_files'] += original_count
        stats['filtered_files'] += filtered_count
        if filter_config.get("enabled", False) and original_count > filtered_count:
            print(f"    Filter: {original_count} -> {filtered_count} trajectories "
                  f"for {sim_config['name']} {temp_str}")
        if not files:
            continue

        output_subdir = _paths.substitute_temp(sim_config['output_subdir'], temp)
        out_dir = os.path.join(paths_config['cg_data_base_dir'], output_subdir)
        if processing_config.get('create_output_dirs', True):
            ensure_dir(out_dir)

        tasks = [(f, out_dir, cg_config, output_config) for f in files]
        max_workers = processing_config.get('max_workers')
        parallel = processing_config.get('parallel', True)

        results = _parallel.run_parallel(
            tasks, _process_one_case,
            n_workers=max_workers, parallel=parallel,
            desc=f"  {sim_config['name']} {temp_str}", unit="file",
        )
        for in_file, ok, msg, file_stats in results:
            if ok:
                stats['success'] += 1
                stats['total_timesteps'] += file_stats.get('timesteps', 0)
                stats['total_particles'] += file_stats.get('particles', 0)
            else:
                stats['failed'] += 1
                stats['errors'].append(f"{os.path.basename(in_file)}: {msg}")

    return stats


# =============================================================================
# Entry point (replaces legacy main())
# =============================================================================

def run(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """cgkit cg-gen entry point."""
    paths_config = config['paths']
    cg_config = config['coarse_graining']
    output_config = config.get('output', {})
    processing_config = config.get('processing', {})

    print(f"\n{'=' * 60}")
    print("COARSE-GRAINING DATA PROCESSING")
    print(f"{'=' * 60}")
    print(f"CG Method: {cg_config['method']}")
    if cg_config['method'] == 'flexible_pattern':
        patterns = cg_config.get('patterns', [])
        pattern_to_type: Dict[Tuple[int, ...], int] = {}
        type_counter = 1
        for pattern in patterns:
            tu = tuple(pattern)
            if tu not in pattern_to_type:
                pattern_to_type[tu] = type_counter
                type_counter += 1
        print("Pattern -> CG Type mapping:")
        for tu, cg_type in sorted(pattern_to_type.items(), key=lambda x: x[1]):
            print(f"  {list(tu)} -> Type {cg_type} ({len(tu)} atoms)")
        print(f"Center atom type: {cg_config.get('center_atom_type', 1)}")
        cg_assignments = cg_config.get('cg_assignments', [])
        if cg_assignments:
            print("Manual atom assignments:")
            for a in cg_assignments:
                ids = a.get('atom_ids', [])
                t = a.get('cg_type', 1)
                desc = a.get('description', '')
                print(f"  Atom IDs {ids} -> CG Type {t} ({len(ids)} atoms)"
                      + (f" ({desc})" if desc else ""))

    all_stats = []
    for sim_config in config.get('simulations', []):
        if not sim_config.get('enabled', True):
            continue
        print(f"\nProcessing {sim_config['name']}...")
        stats = process_simulation(sim_config, paths_config, cg_config,
                                   output_config, processing_config)
        all_stats.append(stats)
        print(f"  Files: {stats['total_files']} (filtered: {stats['filtered_files']})")
        print(f"  Success: {stats['success']}, Failed: {stats['failed']}")
        print(f"  Total timesteps: {stats['total_timesteps']}, "
              f"Total particles: {stats['total_particles']}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for s in all_stats:
        print(f"{s['name']}: success={s['success']} failed={s['failed']} "
              f"timesteps={s['total_timesteps']} particles={s['total_particles']}")
        if s['errors']:
            print(f"  Errors:")
            for err in s['errors'][:5]:
                print(f"    {err}")
            if len(s['errors']) > 5:
                print(f"    ... and {len(s['errors']) - 5} more")

    total_failed = sum(s['failed'] for s in all_stats)
    return 0 if total_failed == 0 else 1
