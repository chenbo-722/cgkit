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


def _unwrap_chain_coords(df: pd.DataFrame,
                         box_bounds: List[List[float]]) -> pd.DataFrame:
    """按 id 顺序做链式 PBC unwrap，原地覆写 x/y/z（同步 xu/yu/zu）。

    对每个原子 i>0（按 id 升序），若与前一原子在某轴的位移超过半个盒长，
    则 ±L 折叠，使整条链在物理空间连续。PE 等线性共价体系下等价于
    重建分子链；对均匀液体（无链序）可能产生伪连续，应通过
    ``coarse_graining.unwrap_pbc=false`` 关闭。

    返回传入的 df（已就地修改），便于链式调用。
    """
    if df is None or len(df) == 0:
        return df
    Lx = box_bounds[0][1] - box_bounds[0][0]
    Ly = box_bounds[1][1] - box_bounds[1][0]
    Lz = box_bounds[2][1] - box_bounds[2][0]
    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        return df  # 盒子异常，跳过

    if 'id' in df.columns:
        order = df.sort_values('id').index.tolist()
    else:
        order = df.index.tolist()

    for col, L in (('x', Lx), ('y', Ly), ('z', Lz)):
        if col not in df.columns:
            continue
        vals = df.loc[order, col].astype(float).values.copy()
        for i in range(1, len(vals)):
            d = vals[i] - vals[i - 1]
            vals[i] -= round(d / L) * L
        df.loc[order, col] = vals

    # 同步 xu/yu/zu（若存在），保证下游所有读取路径一致
    for ucol, col in (('xu', 'x'), ('yu', 'y'), ('zu', 'z')):
        if ucol in df.columns and col in df.columns:
            df[ucol] = df[col]

    return df


# =============================================================================
# Coarse-graining
# =============================================================================

def coarse_grain_trajectory(parser: LammpsDumpReader,
                            timestep_index: int = -1,
                            cg_config: Optional[Dict[str, Any]] = None
                            ) -> Tuple[Optional[List[Dict]], Optional[List[float]], List[str]]:
    """灵活模式粗粒化。

    返回 ``(coarse_particles, box_vector, warnings)``。``warnings`` 收集
    id_pattern 匹配失败等非致命问题，由上层汇总打印。
    """
    frame_warnings: List[str] = []
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
        return None, None, frame_warnings

    if timestep_index < 0:
        timestep_index = len(parser.timesteps) + timestep_index
    if timestep_index >= len(parser.timesteps):
        return None, None, frame_warnings

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

    # 新参数：PBC 链式 unwrap、r_cutoff、id_patterns
    unwrap_pbc = cg_config.get("unwrap_pbc", False)
    r_cutoff = cg_config.get("r_cutoff", None)
    id_patterns = cg_config.get("id_patterns", []) or []

    if use_unwrapped and not has_unwrapped:
        use_unwrapped = False

    # 链式 unwrap：把 wrapped 坐标重建为物理连续坐标（覆写 x/y/z 与 xu/yu/zu）
    if unwrap_pbc:
        df = _unwrap_chain_coords(df, box_bounds)
        # unwrap 后 use_unwrapped 应为 True，让 calculate_distance / get_position
        # 走 xu/yu/zu 路径（值已被同步为 unwrap）
        use_unwrapped = True
        has_unwrapped = True
        # 强制 df 含 xu/yu/zu 列
        for ucol, col in (('xu', 'x'), ('yu', 'y'), ('zu', 'z')):
            if ucol not in df.columns and col in df.columns:
                df[ucol] = df[col]

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
                # Mass-weighted force averaging: type 1 (C)=12, type 2 (H)=1
                _type_mass = {1: 12.0, 2: 1.0}
                masses = atom_rows['type'].map(_type_mass)
                total_mass = masses.sum()
                avg_fx = (atom_rows['fx'] * masses).sum() / total_mass
                avg_fy = (atom_rows['fy'] * masses).sum() / total_mass
                avg_fz = (atom_rows['fz'] * masses).sum() / total_mass
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

    # Step 1: manual assignments (highest priority)
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
        particle['match_status'] = 'manual'
        coarse_particles.append(particle)
        used_atom_indices.update(target_atoms.index)
        cg_id += 1

    # Step 1.5: id_patterns (id-offset based binding)
    # 优先级介于 cg_assignments 与 patterns 之间；匹配失败的中心原子留给
    # 后续 patterns 兜底（不占用 center_row），仅记录 warning。
    if id_patterns and 'id' in df.columns:
        id_to_idx = {int(v): i for i, v in enumerate(df['id'].astype(int).values)}
        for idp in id_patterns:
            type_pattern = idp.get("type_pattern", []) or []
            id_offsets = idp.get("id_offsets", []) or []
            assigned_cg_type = idp.get("cg_type", 1)

            # 完整性校验
            if len(type_pattern) < 2 or len(type_pattern) != len(id_offsets):
                frame_warnings.append(
                    f"id_pattern 跳过：type_pattern 与 id_offsets 长度不一致或过短: {idp}")
                continue
            if id_offsets[0] != 0:
                frame_warnings.append(
                    f"id_pattern 跳过：中心偏移必须为 0， got id_offsets={id_offsets}")
                continue
            if type_pattern[0] != center_type:
                # 不是当前 center_type 的规则，跳过（不算错误）
                continue

            sub_offsets = id_offsets[1:]
            sub_types = type_pattern[1:]

            # 候选中心原子：当前类型 + 未被占用
            center_candidates = df[
                (df['type'] == center_type)
                & (~df.index.isin(used_atom_indices))
            ]

            for _, center_row in center_candidates.iterrows():
                if center_row.name in used_atom_indices:
                    continue
                center_id = int(center_row['id'])

                sub_indices: List[int] = []
                match_ok = True
                for off, exp_type in zip(sub_offsets, sub_types):
                    target_id = center_id + off
                    if target_id not in id_to_idx:
                        frame_warnings.append(
                            f"id_pattern: 中心 id={center_id} 缺 id={target_id} (offset={off})")
                        match_ok = False
                        break
                    target_idx = id_to_idx[target_id]
                    target_type = int(df.loc[target_idx, 'type'])
                    if target_type != exp_type:
                        frame_warnings.append(
                            f"id_pattern: 中心 id={center_id}, id={target_id} "
                            f"type={target_type} != 期望 {exp_type}")
                        match_ok = False
                        break
                    if target_idx in used_atom_indices:
                        frame_warnings.append(
                            f"id_pattern: 中心 id={center_id}, id={target_id} 已被占用")
                        match_ok = False
                        break
                    sub_indices.append(target_idx)

                if not match_ok:
                    # 不占用中心原子，留给 patterns 兜底
                    continue

                selected_indices = [center_row.name] + sub_indices
                atom_rows = df.loc[selected_indices]
                particle = create_cg_particle(atom_rows, assigned_cg_type)
                particle['manual_assignment'] = False
                particle['match_status'] = 'id_pattern'
                particle['id_pattern'] = {
                    'type_pattern': list(type_pattern),
                    'id_offsets': list(id_offsets),
                }
                coarse_particles.append(particle)
                used_atom_indices.update(atom_rows.index)
                cg_id += 1
    elif id_patterns and 'id' not in df.columns:
        frame_warnings.append("id_pattern 跳过：原子 dump 无 id 列")

    # Step 2: pattern-based (distance) for remaining atoms
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
        return [], box_vector, frame_warnings

    cutoff_sq = float(r_cutoff) ** 2 if r_cutoff is not None else None

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

            # r_cutoff 过滤：仅保留截止半径内的候选
            if cutoff_sq is not None:
                distances = [t for t in distances if t[0] <= cutoff_sq]

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
            # 所有 pattern 都因 r_cutoff 不足或候选不够而失败
            frame_warnings.append(
                f"中心原子 id={int(center_row.get('id', -1))} 无匹配 pattern "
                f"(r_cutoff={r_cutoff})")
            continue

        pattern_tuple = tuple(best_pattern)
        cg_type = pattern_to_cg_type.get(pattern_tuple, 1)
        particle = create_cg_particle(best_match_atoms, cg_type)
        particle['pattern'] = best_pattern
        particle['manual_assignment'] = False
        particle['match_status'] = 'pattern'
        coarse_particles.append(particle)
        used_atom_indices.update(best_match_atoms.index)
        cg_id += 1

    return coarse_particles, box_vector, frame_warnings


def process_all_timesteps(parser: LammpsDumpReader,
                          cg_config: Optional[Dict[str, Any]] = None
                          ) -> Tuple[List[Dict], np.ndarray, List[str]]:
    """处理所有时间步。

    返回 ``(all_coarse_data, all_box_vectors, all_warnings)``。
    """
    all_coarse_data: List[Dict[str, Any]] = []
    all_box_vectors: List[List[float]] = []
    all_warnings: List[str] = []
    for timestep_idx in range(len(parser.timesteps)):
        coarse_particles, box_vector, frame_warnings = coarse_grain_trajectory(
            parser, timestep_idx, cg_config)
        if coarse_particles is not None:
            for particle in coarse_particles:
                particle['timestep'] = parser.timesteps[timestep_idx]
            all_coarse_data.extend(coarse_particles)
            all_box_vectors.append(box_vector)
        all_warnings.extend(frame_warnings)
    return all_coarse_data, np.array(all_box_vectors), all_warnings


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
    """Process a single trajectory file."""
    parser = LammpsDumpReader(filename)
    if not parser.parse_file(verbose=False):
        raise Exception("File parsing failed!")
    all_coarse_data, all_box_vectors, all_warnings = process_all_timesteps(parser, cg_config)
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
        'warnings': all_warnings,
    }


def _process_one_case(task: Tuple[str, str, Dict[str, Any], Dict[str, Any]]
                      ) -> Tuple[bool, str, Dict[str, Any]]:
    """Top-level worker for ProcessPoolExecutor (must be picklable).

    Returns the ``run_parallel`` 3-tuple ``(ok, msg, stats)`` — the task
    payload itself is prepended by the caller, so we must NOT include
    ``in_file`` here (doing so caused a tuple-unpacking cascade).
    """
    in_file, out_dir, cg_config, output_config = task
    try:
        stats = traj2CG(in_file, out_dir, cg_config, output_config)
        return True, '', stats
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), {}


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
        'warnings': [],
    }
    if not stats['enabled']:
        return stats

    traj_dir_template = os.path.join(paths_config['base_dir'], sim_config['trajectory_dir'])

    filter_config = processing_config.get('trajectory_filter', {}) or {}
    temperatures = sim_config.get('temperatures') or [None]

    for temp in temperatures:
        temp_str = f"{temp}K" if temp is not None else "N/A"
        # Substitute {temp} into both the directory and the glob pattern, so
        # layouts like "<sim>/<temp>/traj/" with per-temp subdirectories work.
        # (Existence check is done per-temp inside the loop; an early exit on
        # a literal "{temp}" template used to silently skip such sims.)
        traj_dir = _paths.substitute_temp(traj_dir_template, temp)
        if not os.path.exists(traj_dir):
            stats['errors'].append(
                f"Trajectory directory does not exist for {sim_config['name']} "
                f"{temp_str}: {traj_dir}")
            continue
        files = glob_with_temp(traj_dir_template, sim_config['file_pattern'], temp)
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
        for task, ok, msg, file_stats in results:
            if ok:
                stats['success'] += 1
                stats['total_timesteps'] += file_stats.get('timesteps', 0)
                stats['total_particles'] += file_stats.get('particles', 0)
                file_warnings = file_stats.get('warnings') or []
                if file_warnings:
                    in_file = task[0]
                    basename = os.path.basename(in_file)
                    for w in file_warnings:
                        stats['warnings'].append(f"{basename}: {w}")
            else:
                stats['failed'] += 1
                in_file = task[0]  # task = (in_file, out_dir, cg_config, output_config)
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

    # CLI overrides for new algorithm knobs
    if hasattr(args, 'r_cutoff') and args.r_cutoff is not None:
        cg_config['r_cutoff'] = args.r_cutoff
    if hasattr(args, 'unwrap_pbc') and args.unwrap_pbc is not None:
        cg_config['unwrap_pbc'] = args.unwrap_pbc

    print(f"\n{'=' * 60}")
    print("COARSE-GRAINING DATA PROCESSING")
    print(f"{'=' * 60}")
    print(f"CG Method: {cg_config['method']}")
    print(f"PBC chain unwrap: {cg_config.get('unwrap_pbc', False)}")
    print(f"r_cutoff (pattern distance): {cg_config.get('r_cutoff', None)}")
    if cg_config.get('id_patterns'):
        print(f"id_patterns: {len(cg_config['id_patterns'])} rule(s)")
        for idp in cg_config['id_patterns']:
            desc = idp.get('description', '')
            print(f"  type_pattern={idp.get('type_pattern')} "
                  f"id_offsets={idp.get('id_offsets')} "
                  f"cg_type={idp.get('cg_type', 1)}"
                  + (f" ({desc})" if desc else ""))
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

    # Warnings aggregation (dedupe by message template, count occurrences)
    all_warnings: List[str] = []
    for s in all_stats:
        all_warnings.extend(s.get('warnings', []))
    if all_warnings:
        # Group by warning message (strip leading "filename: " for dedup)
        from collections import OrderedDict
        grouped: 'OrderedDict[str, List[str]]' = OrderedDict()
        for w in all_warnings:
            if ': ' in w:
                prefix, msg = w.split(': ', 1)
            else:
                prefix, msg = '', w
            grouped.setdefault(msg, []).append(prefix)
        print(f"\nWARNINGS ({len(all_warnings)} total, {len(grouped)} unique):")
        for msg, files in list(grouped.items())[:20]:
            sample = files[0]
            extra = f" (+{len(files) - 1} more files)" if len(files) > 1 else ""
            print(f"  [{sample}{extra}] {msg}")
        if len(grouped) > 20:
            print(f"  ... and {len(grouped) - 20} more unique warning types")

    total_failed = sum(s['failed'] for s in all_stats)
    return 0 if total_failed == 0 else 1
