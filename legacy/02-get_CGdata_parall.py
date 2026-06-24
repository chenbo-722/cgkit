# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import concurrent.futures
from IPython.display import display, HTML
import pandas as pd
import glob
import re
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

# Set matplotlib to display in Notebook
#%matplotlib inline

# Quiet mode for pandas
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration Management
# =============================================================================

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file

    Args:
        config_path: Path to JSON config file. If None, uses default path.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge command-line arguments with JSON config (CLI args take precedence)

    Args:
        config: Configuration dictionary from JSON
        args: Parsed command-line arguments

    Returns:
        Merged configuration dictionary
    """
    if hasattr(args, 'config') and args.config:
        # Config file specified in CLI, reload it
        config = load_config(args.config)

    if hasattr(args, 'sim') and args.sim is not None:
        # Filter simulations to only the specified one
        config['simulations'] = [s for s in config['simulations'] if s['name'] == args.sim]

    if hasattr(args, 'temp') and args.temp is not None:
        # Override temperatures for all simulations
        for sim in config['simulations']:
            if sim['temperatures'] is not None:
                sim['temperatures'] = args.temp

    if hasattr(args, 'base_dir') and args.base_dir:
        config['paths']['base_dir'] = args.base_dir

    if hasattr(args, 'output_dir') and args.output_dir:
        config['paths']['output_base_dir'] = args.output_dir

    if hasattr(args, 'workers') and args.workers is not None:
        config['processing']['max_workers'] = args.workers

    return config


# =============================================================================
# LAMMPS Trajectory Parser
# =============================================================================

class LAMMPSTrajectoryParser:
    def __init__(self, filename: str):
        self.filename = filename
        self.timesteps = []
        self.natoms = []
        self.box_bounds = []
        self.atoms_data = []

    def parse_file(self, verbose: bool = False) -> bool:
        """Parse LAMMPS trajectory file"""
        if verbose:
            print(f"Parsing file: {self.filename}")

        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if line == "ITEM: TIMESTEP":
                    i += 1
                    timestep = int(lines[i].strip())
                    self.timesteps.append(timestep)

                elif line == "ITEM: NUMBER OF ATOMS":
                    i += 1
                    natoms = int(lines[i].strip())
                    self.natoms.append(natoms)

                elif line.startswith("ITEM: BOX BOUNDS"):
                    i += 1
                    box = []
                    for j in range(3):
                        bounds = list(map(float, lines[i+j].split()))
                        box.append(bounds)
                    i += 2
                    self.box_bounds.append(box)

                elif line.startswith("ITEM: ATOMS"):
                    columns = line.split()[2:]
                    i += 1
                    atoms = []
                    for j in range(self.natoms[-1]):
                        if i+j >= len(lines):
                            break
                        atom_data = list(map(float, lines[i+j].split()))
                        atom_dict = {col: val for col, val in zip(columns, atom_data)}
                        atoms.append(atom_dict)
                    i += self.natoms[-1] - 1
                    self.atoms_data.append(atoms)

                i += 1

            if verbose:
                print(f"Successfully parsed {len(self.timesteps)} timesteps")
            return True

        except Exception as e:
            if verbose:
                print(f"Error parsing file: {str(e)}")
            return False

    def display_info(self, timestep_index: int = -1):
        """Display information for specified timestep"""
        if not self.timesteps:
            print("No valid data found")
            return

        if timestep_index < 0:
            timestep_index = len(self.timesteps) + timestep_index

        if timestep_index >= len(self.timesteps):
            print(f"Timestep index {timestep_index} out of range")
            return

        info_html = f"""
        <p><b>Information for Timestep {self.timesteps[timestep_index]}:</b></p>
        <p><b>Number of atoms:</b> {self.natoms[timestep_index]}</p>
        <p><b>Box boundaries:</b></p>
        <ul>
            <li>X: {self.box_bounds[timestep_index][0][0]:.6f} to {self.box_bounds[timestep_index][0][1]:.6f}</li>
            <li>Y: {self.box_bounds[timestep_index][1][0]:.6f} to {self.box_bounds[timestep_index][1][1]:.6f}</li>
            <li>Z: {self.box_bounds[timestep_index][2][0]:.6f} to {self.box_bounds[timestep_index][2][1]:.6f}</li>
        </ul>
        """

        display(HTML(info_html))

        atoms = self.atoms_data[timestep_index]
        for i in range(min(5, len(atoms))):
            atom = atoms[i]
            print(f"Atom {i+1}: ID={atom.get('id', 'N/A')}, Type={atom.get('type', 'N/A')}")
            print(f"  Position: x={atom.get('x', atom.get('xu', 'N/A')):.4f}, "
                  f"y={atom.get('y', atom.get('yu', 'N/A')):.4f}, "
                  f"z={atom.get('z', atom.get('zu', 'N/A')):.4f}")
            print(f"  Potential energy: {atom.get('c_pe', 'N/A'):.6f}")
            print(f"  Force: fx={atom.get('fx', 'N/A'):.6f}, "
                  f"fy={atom.get('fy', 'N/A'):.6f}, "
                  f"fz={atom.get('fz', 'N/A'):.6f}")

    def get_dataframe(self, timestep_index: int = -1) -> Optional[pd.DataFrame]:
        """Convert data for specified timestep to Pandas DataFrame"""
        if not self.atoms_data:
            print("No data available for conversion")
            return None

        if timestep_index < 0:
            timestep_index = len(self.timesteps) + timestep_index

        if timestep_index >= len(self.timesteps):
            print(f"Timestep index {timestep_index} out of range")
            return None

        atoms = self.atoms_data[timestep_index]
        return pd.DataFrame(atoms)


# =============================================================================
# Coarse-Graining Functions
# =============================================================================

def calculate_distance(pos1: Dict[str, float], pos2: Dict[str, float],
                       box_bounds: List[List[float]], use_unwrapped: bool = True) -> float:
    """
    Calculate distance between two atoms, considering periodic boundary conditions

    Args:
        pos1: First atom position dict with 'x', 'y', 'z' or 'xu', 'yu', 'zu'
        pos2: Second atom position dict
        box_bounds: Box boundaries [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
        use_unwrapped: Whether to use unwrapped coordinates

    Returns:
        Distance squared
    """
    # Get coordinates
    if use_unwrapped:
        x1, y1, z1 = pos1.get('xu', pos1.get('x', 0)), pos1.get('yu', pos1.get('y', 0)), pos1.get('zu', pos1.get('z', 0))
        x2, y2, z2 = pos2.get('xu', pos2.get('x', 0)), pos2.get('yu', pos2.get('y', 0)), pos2.get('zu', pos2.get('z', 0))
    else:
        x1, y1, z1 = pos1.get('x', 0), pos1.get('y', 0), pos1.get('z', 0)
        x2, y2, z2 = pos2.get('x', 0), pos2.get('y', 0), pos2.get('z', 0)

    # Box dimensions
    lx = box_bounds[0][1] - box_bounds[0][0]
    ly = box_bounds[1][1] - box_bounds[1][0]
    lz = box_bounds[2][1] - box_bounds[2][0]

    # Minimum image convention
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    dx = dx - round(dx / lx) * lx
    dy = dy - round(dy / ly) * ly
    dz = dz - round(dz / lz) * lz

    return dx*dx + dy*dy + dz*dz


def coarse_grain_trajectory(parser: LAMMPSTrajectoryParser,
                           timestep_index: int = -1,
                           cg_config: Dict[str, Any] = None) -> Tuple[List[Dict], List[float]]:
    """
    对LAMMPS轨迹数据进行粗粒化处理 - 灵活模式方法

    Args:
        parser: LAMMPSTrajectoryParser实例
        timestep_index: 要处理的时间步索引
        cg_config: 粗粒化配置字典，包含 patterns 配置

    Returns:
        (coarse_particles, box_vector): 粗粒化粒子列表和盒子向量
    """
    if cg_config is None:
        cg_config = {
            "method": "flexible_pattern",
            "patterns": [[1, 2, 2], [1, 2, 2, 2]],
            "center_atom_type": 1,
            "position_source": "unwrapped",
            "average_forces": True,
            "average_potential_energy": True
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

    # Get configuration
    patterns = cg_config.get("patterns", [[1, 2, 2]])
    center_type = cg_config.get("center_atom_type", 1)
    position_source = cg_config.get("position_source", "unwrapped")
    use_unwrapped = (position_source == "unwrapped")
    has_unwrapped = all(col in df.columns for col in ['xu', 'yu', 'zu'])
    cg_assignments = cg_config.get("cg_assignments", [])

    if use_unwrapped and not has_unwrapped:
        use_unwrapped = False

    coarse_particles = []
    cg_id = 1
    used_atom_indices = set()  # Track atoms that have been assigned

    # Helper function to get position from a row
    def get_position(row):
        pos = {}
        if 'xu' in df.columns and 'yu' in df.columns and 'zu' in df.columns:
            pos['xu'] = row['xu']
            pos['yu'] = row['yu']
            pos['zu'] = row['zu']
            pos['x'] = row.get('x', row['xu'])
            pos['y'] = row.get('y', row['yu'])
            pos['z'] = row.get('z', row['zu'])
        elif 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            pos['x'] = row['x']
            pos['y'] = row['y']
            pos['z'] = row['z']
            pos['xu'] = row['x']
            pos['yu'] = row['y']
            pos['zu'] = row['z']
        return pos

    # Helper function to create CG particle
    def create_cg_particle(atom_rows, cg_type, center_idx=0):
        nonlocal cg_id

        # Get force/energy source setting
        force_energy_source = cg_config.get("force_energy_source", "average")

        # Get center atom for position and possibly for force/energy
        center_row = atom_rows.iloc[center_idx]
        pos = get_position(center_row)

        if use_unwrapped:
            pos_x, pos_y, pos_z = pos['xu'], pos['yu'], pos['zu']
        else:
            pos_x, pos_y, pos_z = pos['x'], pos['y'], pos['z']

        # Calculate forces and potential energy based on source setting
        if force_energy_source == "center_only":
            # Use only center atom's force and energy
            avg_fx = center_row['fx']
            avg_fy = center_row['fy']
            avg_fz = center_row['fz']
            avg_pe = center_row['c_pe'] if 'c_pe' in df.columns else None
        else:
            # Use averaged force and energy (default behavior, respects average_forces setting)
            if cg_config.get("average_forces", True):
                avg_fx = atom_rows['fx'].mean()
                avg_fy = atom_rows['fy'].mean()
                avg_fz = atom_rows['fz'].mean()
            else:
                avg_fx = 0
                avg_fy = 0
                avg_fz = 0

            avg_pe = None
            if cg_config.get("average_potential_energy", True) and 'c_pe' in df.columns:
                avg_pe = atom_rows['c_pe'].mean()

        return {
            'id': cg_id,
            'type': cg_type,
            'x': pos_x,
            'y': pos_y,
            'z': pos_z,
            'fx': avg_fx,
            'fy': avg_fy,
            'fz': avg_fz,
            'c_pe': avg_pe,
            'n_atoms': len(atom_rows),
            'atom_indices': list(atom_rows.index)
        }

    # ========================================
    # Step 1: Process manual assignments (cg_assignments)
    # ========================================
    for assignment in cg_assignments:
        atom_ids = assignment.get("atom_ids", [])
        assigned_cg_type = assignment.get("cg_type", 1)

        # Find atoms by ID (use 'id' column or index)
        target_atoms = None
        if 'id' in df.columns:
            target_atoms = df[df['id'].isin(atom_ids)]
        else:
            # If no 'id' column, treat atom_ids as indices (1-based to 0-based)
            target_indices = [aid - 1 for aid in atom_ids if aid - 1 < len(df)]
            target_atoms = df.loc[target_indices]

        if target_atoms is None or len(target_atoms) == 0:
            continue

        # Check if any atoms are already used
        if any(idx in used_atom_indices for idx in target_atoms.index):
            continue  # Skip if any atom is already assigned

        # Create CG particle
        particle = create_cg_particle(target_atoms, assigned_cg_type)
        particle['manual_assignment'] = True
        particle['assigned_atom_ids'] = atom_ids
        coarse_particles.append(particle)
        used_atom_indices.update(target_atoms.index)
        cg_id += 1

    # ========================================
    # Step 2: Process remaining atoms with pattern-based method
    # ========================================

    # Build pattern type mapping: pattern -> cg_type
    # Different atom counts get different types
    pattern_to_cg_type = {}
    cg_type_counter = 1
    for pattern in patterns:
        pattern_tuple = tuple(pattern)
        if pattern_tuple not in pattern_to_cg_type:
            pattern_to_cg_type[pattern_tuple] = cg_type_counter
            cg_type_counter += 1

    # Find remaining center atoms (type = center_atom_type, not used)
    available_df = df[~df.index.isin(used_atom_indices)]
    center_atoms = available_df[available_df['type'] == center_type]

    if len(center_atoms) == 0 and len(coarse_particles) == 0:
        return [], box_vector

    # For each remaining center atom, find matching pattern
    for _, center_row in center_atoms.iterrows():
        # Skip if already used
        if center_row.name in used_atom_indices:
            continue

        center_pos = get_position(center_row)
        if not center_pos:
            continue

        # Try each pattern to see which one fits best
        best_pattern = None
        best_match_atoms = None
        best_score = -1

        for pattern in patterns:
            pattern_length = len(pattern)
            n_neighbor_type2 = pattern.count(2)

            # Find available type2 atoms (not used yet)
            available_type2 = available_df[(available_df['type'] == 2) & (~available_df.index.isin(used_atom_indices))]

            if len(available_type2) < n_neighbor_type2:
                continue

            # Calculate distances to all available type2 atoms
            distances = []
            for idx, type2_row in available_type2.iterrows():
                type2_pos = get_position(type2_row)
                if not type2_pos:
                    continue
                dist_sq = calculate_distance(center_pos, type2_pos, box_bounds, use_unwrapped)
                distances.append((dist_sq, idx))

            if len(distances) < n_neighbor_type2:
                continue

            # Sort by distance and pick nearest n_neighbor_type2 atoms
            distances.sort(key=lambda x: x[0])
            selected_type2_indices = [idx for _, idx in distances[:n_neighbor_type2]]

            selected_atoms = [center_row.name] + selected_type2_indices
            selected_atom_rows = df.loc[selected_atoms]

            # Calculate score: pattern fit (prefer shorter patterns if enough atoms available)
            score = len(selected_atoms) - pattern_length * 0.01

            if score > best_score:
                best_score = score
                best_pattern = pattern
                best_match_atoms = selected_atom_rows

        # If no pattern matched, skip this center atom
        if best_match_atoms is None:
            continue

        # Determine CG type based on pattern
        pattern_tuple = tuple(best_pattern)
        cg_type = pattern_to_cg_type.get(pattern_tuple, 1)

        # Create CG particle using helper function
        particle = create_cg_particle(best_match_atoms, cg_type)
        particle['pattern'] = best_pattern
        coarse_particles.append(particle)
        used_atom_indices.update(best_match_atoms.index)
        cg_id += 1

    return coarse_particles, box_vector


def process_all_timesteps(parser: LAMMPSTrajectoryParser,
                         cg_config: Dict[str, Any] = None) -> Tuple[List[Dict], np.ndarray]:
    """
    处理所有时间步的粗粒化数据

    Args:
        parser: LAMMPSTrajectoryParser实例
        cg_config: 粗粒化配置字典

    Returns:
        (all_coarse_data, all_box_vectors): 所有时间步的粗粒化数据和盒子向量
    """
    all_coarse_data = []
    all_box_vectors = []

    for timestep_idx in range(len(parser.timesteps)):
        coarse_particles, box_vector = coarse_grain_trajectory(parser, timestep_idx, cg_config)

        if coarse_particles is not None:
            for particle in coarse_particles:
                particle['timestep'] = parser.timesteps[timestep_idx]
            all_coarse_data.extend(coarse_particles)
            all_box_vectors.append(box_vector)

    return all_coarse_data, np.array(all_box_vectors)


# =============================================================================
# Data Export Functions
# =============================================================================

def export_coarse_grained_data(parser: LAMMPSTrajectoryParser,
                               coarse_df: pd.DataFrame,
                               box_vectors: np.ndarray,
                               output_prefix: str,
                               output_config: Dict[str, Any] = None,
                               verbose: bool = False):
    """
    导出粗粒化数据

    Args:
        parser: LAMMPSTrajectoryParser实例
        coarse_df: 粗粒化粒子数据DataFrame
        box_vectors: 盒子向量数组
        output_prefix: 输出文件前缀
        output_config: 输出配置字典
        verbose: 是否输出详细信息
    """
    if output_config is None:
        output_config = {
            "save_particles": True,
            "save_box_vectors": True,
            "particles_filename": "{basename}_particles.csv",
            "box_vectors_filename": "{basename}_box_vectors.csv"
        }

    basename = os.path.basename(output_prefix)

    if output_config.get("save_particles", True):
        particle_filename = output_config.get("particles_filename", "{basename}_particles.csv")
        particle_filename = particle_filename.format(basename=basename)
        particle_path = os.path.join(os.path.dirname(output_prefix), particle_filename)
        coarse_df.to_csv(particle_path, index=False)
        if verbose:
            print(f"  Particles data: {particle_filename}")

    if output_config.get("save_box_vectors", True):
        box_filename = output_config.get("box_vectors_filename", "{basename}_box_vectors.csv")
        box_filename = box_filename.format(basename=basename)
        box_path = os.path.join(os.path.dirname(output_prefix), box_filename)
        box_df = pd.DataFrame(box_vectors, columns=[
            'xlo', 'xhi', 'xy',
            'ylo', 'yhi', 'xz',
            'zlo', 'zhi', 'yz'
        ])
        box_df['timestep'] = parser.timesteps[:len(box_vectors)]
        box_df.to_csv(box_path, index=False)
        if verbose:
            print(f"  Box vectors: {box_filename}")


def export_cg_trajectory(parser: LAMMPSTrajectoryParser,
                        all_coarse_data: List[Dict],
                        all_box_vectors: np.ndarray,
                        output_path: str,
                        cg_config: Dict[str, Any] = None):
    """
    Export coarse-grained trajectory in LAMMPS dump format

    Args:
        parser: LAMMPSTrajectoryParser instance
        all_coarse_data: List of all coarse-grained particles with timestep info
        all_box_vectors: Box vectors for each timestep
        output_path: Output file path
        cg_config: Coarse-graining configuration
    """
    if cg_config is None:
        cg_config = {}

    # Group coarse particles by timestep
    timesteps = sorted(set(p['timestep'] for p in all_coarse_data))
    timestep_to_particles = {ts: [p for p in all_coarse_data if p['timestep'] == ts] for ts in timesteps}

    position_source = cg_config.get("position_source", "unwrapped")
    use_unwrapped = (position_source == "unwrapped")

    with open(output_path, 'w') as f:
        for ts_idx, timestep in enumerate(timesteps):
            particles = timestep_to_particles[timestep]
            box_vector = all_box_vectors[ts_idx]

            # Reconstruct box_bounds from box_vector
            xlo, xhi = 0, box_vector[0]
            ylo, yhi = 0, box_vector[4]
            zlo, zhi = 0, box_vector[8]

            # Write timestep header
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")

            # Write number of atoms
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(particles)}\n")

            # Write box bounds
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{xlo:.6f} {xhi:.6f} 0.0\n")
            f.write(f"{ylo:.6f} {yhi:.6f} 0.0\n")
            f.write(f"{zlo:.6f} {zhi:.6f} 0.0\n")

            # Write atoms
            if use_unwrapped:
                f.write("ITEM: ATOMS id type xu yu zu fx fy fz\n")
            else:
                f.write("ITEM: ATOMS id type x y z fx fy fz\n")

            for p in sorted(particles, key=lambda x: x['id']):
                # Use the type from particle data (different for different patterns)
                cg_type = p.get('type', 1)
                f.write(f"{p['id']} {cg_type} ")
                if use_unwrapped:
                    f.write(f"{p['x']:.6f} {p['y']:.6f} {p['z']:.6f} ")
                else:
                    # For wrapped coordinates, we'd need to wrap them
                    # For now, use the same coordinates
                    f.write(f"{p['x']:.6f} {p['y']:.6f} {p['z']:.6f} ")
                f.write(f"{p['fx']:.6f} {p['fy']:.6f} {p['fz']:.6f}\n")


# =============================================================================
# Trajectory Filtering Functions
# =============================================================================


def extract_timestep_from_filename(filename: str) -> Optional[int]:
    """
    Extract timestep value from LAMMPS trajectory filename

    Args:
        filename: Path to trajectory file

    Returns:
        Timestep value as integer, or None if not found
    """
    basename = os.path.basename(filename)

    # Try to find timestep in common patterns:
    # 1. Pattern: name.timestep.ext (e.g., NPT.200.10000.lammpstrj)
    # 2. Pattern: name_timestep.ext (e.g., dump_10000.lammpstrj)
    # 3. Pattern: name.timestep (e.g., NPT.200.10000)

    # Remove extension
    name_without_ext = os.path.splitext(basename)[0]

    # Split by common delimiters
    parts = re.split(r'[._-]', name_without_ext)

    # Look for numeric parts that could be timesteps (usually large numbers)
    for part in reversed(parts):  # Check from end, timestep often at the end
        if part.isdigit():
            timestep = int(part)
            # Heuristic: timesteps are usually >= 1000 in MD simulations
            if timestep >= 100:
                return timestep

    return None


def select_trajectories_max_interval(files: List[str], max_trajectories: int) -> List[str]:
    """
    Select trajectory files using maximum interval sampling.

    This method selects files to maximize the minimum interval between
    consecutive selected timesteps, ensuring uniform coverage across time.

    Args:
        files: List of trajectory file paths (sorted by timestep)
        max_trajectories: Maximum number of trajectories to select

    Returns:
        Filtered list of trajectory file paths
    """
    if len(files) <= max_trajectories:
        return files

    # Extract timesteps from files
    file_timestep_pairs = []
    for f in files:
        ts = extract_timestep_from_filename(f)
        if ts is not None:
            file_timestep_pairs.append((f, ts))
        else:
            # If no timestep found, use index as fallback
            file_timestep_pairs.append((f, len(file_timestep_pairs)))

    # Sort by timestep
    file_timestep_pairs.sort(key=lambda x: x[1])

    # Select files using maximum interval method
    # Always include first and last files
    n_files = len(file_timestep_pairs)
    if max_trajectories <= 2:
        # Just return first and last
        return [file_timestep_pairs[0][0], file_timestep_pairs[-1][0]]

    # Use k-means like approach for interval maximization
    # Initialize with first and last files
    selected_indices = [0, n_files - 1]

    # Greedily add files that maximize minimum interval
    while len(selected_indices) < max_trajectories:
        best_idx = -1
        best_min_interval = -1

        for i in range(n_files):
            if i in selected_indices:
                continue

            # Calculate intervals if we add this file
            temp_indices = sorted(selected_indices + [i])
            intervals = [temp_indices[j+1] - temp_indices[j] for j in range(len(temp_indices) - 1)]
            min_interval = min(intervals)

            if min_interval > best_min_interval:
                best_min_interval = min_interval
                best_idx = i

        selected_indices.append(best_idx)

    # Sort indices and extract files
    selected_indices.sort()
    selected_files = [file_timestep_pairs[i][0] for i in selected_indices]

    return selected_files


def apply_trajectory_filter(files: List[str],
                            sim_name: str,
                            temp: Optional[int],
                            filter_config: Dict[str, Any]) -> List[str]:
    """
    Apply trajectory filtering based on configuration.

    Args:
        files: List of trajectory file paths
        sim_name: Simulation name (e.g., "1-npt")
        temp: Temperature value (or None for non-temperature sims)
        filter_config: Trajectory filter configuration

    Returns:
        Filtered list of trajectory file paths
    """
    # Check if filtering is enabled
    if not filter_config.get("enabled", False):
        return files

    # Get max_trajectories for this simulation
    per_temp_config = filter_config.get("per_temp", {})

    max_trajectories = None
    method = "max_interval"

    # Priority order for finding max_trajectories:
    # 1. For simulations with temperature: per_temp[sim_name].temperatures[temp]
    # 2. For simulations with temperature: per_temp[sim_name].max_trajectories (fallback)
    # 3. For simulations without temperature: per_temp[sim_name].max_trajectories
    # 4. per_temp.default.max_trajectories
    # 5. If None/0, process all files

    if temp is not None:
        # Temperature-specific simulation
        if sim_name in per_temp_config:
            sim_config = per_temp_config[sim_name]
            # Try per-temperature setting first
            temp_config = sim_config.get("temperatures") or {}
            temp_key = str(temp)
            if temp_key in temp_config:
                max_trajectories = temp_config[temp_key]
                # Also check for method override in temp config
                if isinstance(temp_config[temp_key], dict):
                    max_trajectories = temp_config[temp_key].get("max_trajectories")
                    method = temp_config[temp_key].get("selection_method", method)
            # Fall back to sim-level max_trajectories
            elif "max_trajectories" in sim_config:
                max_trajectories = sim_config["max_trajectories"]

    # For non-temperature sims or if max_trajectories still not found
    if max_trajectories is None:
        if sim_name in per_temp_config and "max_trajectories" in per_temp_config[sim_name]:
            max_trajectories = per_temp_config[sim_name]["max_trajectories"]
        elif "default" in per_temp_config:
            max_trajectories = per_temp_config["default"].get("max_trajectories")

    # Get selection method (from temp config, sim config, or default)
    if temp is not None and sim_name in per_temp_config:
        temp_config = per_temp_config[sim_name].get("temperatures") or {}
        temp_key = str(temp)
        if temp_key in temp_config and isinstance(temp_config[temp_key], dict):
            method = temp_config[temp_key].get("selection_method", method)
    if method == "max_interval" and sim_name in per_temp_config:
        method = per_temp_config[sim_name].get("selection_method", method)
    if method == "max_interval" and "default" in per_temp_config:
        method = per_temp_config["default"].get("selection_method", method)

    # If max_trajectories is None or 0, process all files
    if max_trajectories is None or max_trajectories <= 0:
        return files

    # Apply filtering
    if method == "max_interval":
        return select_trajectories_max_interval(files, max_trajectories)
    else:
        # Default: just take first max_trajectories files
        return files[:max_trajectories]


# =============================================================================
# Main Processing Functions
# =============================================================================

def traj2CG(filename: str, out_dir: str, cg_config: Dict[str, Any],
           output_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single trajectory file

    Args:
        filename: Input trajectory file path
        out_dir: Output directory path
        cg_config: Coarse-graining configuration
        output_config: Output configuration

    Returns:
        Statistics dictionary with timesteps and particles count
    """
    parser = LAMMPSTrajectoryParser(filename)

    if not parser.parse_file(verbose=False):
        raise Exception("File parsing failed!")

    all_coarse_data, all_box_vectors = process_all_timesteps(parser, cg_config)
    coarse_df = pd.DataFrame(all_coarse_data)

    output_prefix = os.path.join(out_dir, os.path.basename(filename))
    export_coarse_grained_data(parser, coarse_df, all_box_vectors, output_prefix, output_config, verbose=False)

    # Export CG trajectory if enabled
    if cg_config.get("export_cg_trajectory", False):
        cg_traj_filename = cg_config.get("cg_trajectory_filename", "{basename}_cg.lammpstrj")
        cg_traj_filename = cg_traj_filename.format(basename=os.path.basename(filename))
        cg_traj_path = os.path.join(out_dir, cg_traj_filename)
        export_cg_trajectory(parser, all_coarse_data, all_box_vectors, cg_traj_path, cg_config)

    return {
        'timesteps': len(all_box_vectors),
        'particles': len(coarse_df)
    }


def _process_one_case(args: Tuple) -> Tuple[str, bool, str, Dict[str, Any]]:
    """Worker function for parallel processing"""
    in_file, out_dir, cg_config, output_config = args
    try:
        stats = traj2CG(in_file, out_dir, cg_config, output_config)
        return in_file, True, '', stats
    except Exception as e:
        return in_file, False, str(e), {}


def scan_trajectory_files(traj_dir: str, pattern: str, temp: Optional[int] = None) -> List[str]:
    """
    扫描指定目录下的轨迹文件

    Args:
        traj_dir: 轨迹目录路径
        pattern: 文件匹配模式
        temp: 温度（可选，用于替换{temp}占位符）

    Returns:
        文件路径列表
    """
    if temp is not None:
        pattern = pattern.replace("{temp}", str(temp))

    search_path = os.path.join(traj_dir, pattern)
    files = glob.glob(search_path)
    files.sort()
    return files


def process_simulation(sim_config: Dict[str, Any],
                      paths_config: Dict[str, Any],
                      cg_config: Dict[str, Any],
                      output_config: Dict[str, Any],
                      processing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个模拟类型的所有轨迹文件

    Args:
        sim_config: 单个模拟的配置
        paths_config: 路径配置
        cg_config: 粗粒化配置
        output_config: 输出配置
        processing_config: 处理配置

    Returns:
        统计信息字典
    """
    stats = {
        'name': sim_config['name'],
        'enabled': sim_config.get("enabled", True),
        'total_files': 0,
        'filtered_files': 0,
        'success': 0,
        'failed': 0,
        'total_timesteps': 0,
        'total_particles': 0,
        'errors': []
    }

    if not stats['enabled']:
        return stats

    # Build input directory path
    traj_dir = os.path.join(paths_config['base_dir'], sim_config['trajectory_dir'])

    if not os.path.exists(traj_dir):
        stats['errors'].append(f"Trajectory directory does not exist: {traj_dir}")
        return stats

    # Get trajectory filter configuration
    filter_config = processing_config.get('trajectory_filter', {})

    # Get temperatures
    temperatures = sim_config.get('temperatures')
    if temperatures is None:
        temperatures = [None]

    for temp in temperatures:
        temp_str = f"{temp}K" if temp is not None else "N/A"

        # Scan files
        files = scan_trajectory_files(traj_dir, sim_config['file_pattern'], temp)

        if not files:
            continue

        # Apply trajectory filtering
        original_count = len(files)
        files = apply_trajectory_filter(files, sim_config['name'], temp, filter_config)
        filtered_count = len(files)
        stats['total_files'] += original_count
        stats['filtered_files'] += filtered_count

        if filter_config.get("enabled", False) and original_count > filtered_count:
            print(f"    Filter: {original_count} -> {filtered_count} trajectories for {sim_config['name']} {temp_str}")

        if not files:
            continue

        # Build output directory
        output_subdir = sim_config['output_subdir']
        if temp is not None:
            output_subdir = output_subdir.replace("{temp}", str(temp))
        out_dir = os.path.join(paths_config['output_base_dir'], output_subdir)

        # Create output directory
        if processing_config.get('create_output_dirs', True):
            os.makedirs(out_dir, exist_ok=True)

        # Build tasks
        tasks = [(f, out_dir, cg_config, output_config) for f in files]

        # Process files with progress bar
        max_workers = processing_config.get('max_workers')
        if max_workers is None:
            max_workers = os.cpu_count() or 1

        if processing_config.get('parallel', True):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                with tqdm(total=len(tasks), desc=f"  {sim_config['name']} {temp_str}",
                          unit='file', ncols=100, leave=False) as pbar:
                    for in_file, ok, msg, file_stats in executor.map(_process_one_case, tasks):
                        if ok:
                            stats['success'] += 1
                            stats['total_timesteps'] += file_stats.get('timesteps', 0)
                            stats['total_particles'] += file_stats.get('particles', 0)
                        else:
                            stats['failed'] += 1
                            stats['errors'].append(f"{os.path.basename(in_file)}: {msg}")
                        pbar.update(1)
        else:
            with tqdm(total=len(tasks), desc=f"  {sim_config['name']} {temp_str}",
                      unit='file', ncols=100, leave=False) as pbar:
                for task in tasks:
                    in_file, ok, msg, file_stats = _process_one_case(task)
                    if ok:
                        stats['success'] += 1
                        stats['total_timesteps'] += file_stats.get('timesteps', 0)
                        stats['total_particles'] += file_stats.get('particles', 0)
                    else:
                        stats['failed'] += 1
                        stats['errors'].append(f"{os.path.basename(in_file)}: {msg}")
                    pbar.update(1)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Coarse-graining data processing for LAMMPS trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config.json
  python get_CGdata_parall.py

  # Use custom config file
  python get_CGdata_parall.py --config my_config.json

  # Process only 1-npt simulations
  python get_CGdata_parall.py --sim 1-npt

  # Process specific temperatures
  python get_CGdata_parall.py --temp 200 300

  # Override base directory
  python get_CGdata_parall.py --base-dir /path/to/simulations

  # Set number of workers
  python get_CGdata_parall.py --workers 4
        """
    )

    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to JSON configuration file (default: config.json)')
    parser.add_argument('--sim', type=str, default=None,
                        choices=['1-npt', '2-nvt', '3-upT', '4-dnT'],
                        help='Process only this simulation type (overrides config)')
    parser.add_argument('--temp', type=int, nargs='+', default=None,
                        help='Temperature(s) to process (overrides config)')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory for simulation data (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output base directory (overrides config)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (overrides config)')

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Merge with command-line arguments
    config = merge_config_with_args(config, args)

    # Extract config sections
    paths_config = config['paths']
    cg_config = config['coarse_graining']
    output_config = config['output']
    processing_config = config['processing']

    # Print header
    print(f"\n{'='*60}")
    print("COARSE-GRAINING DATA PROCESSING")
    print(f"{'='*60}")
    print(f"CG Method: {cg_config['method']}")
    if cg_config['method'] == 'flexible_pattern':
        patterns = cg_config.get('patterns', [])
        # Build pattern-to-type mapping for display
        pattern_to_type = {}
        type_counter = 1
        for pattern in patterns:
            pattern_tuple = tuple(pattern)
            if pattern_tuple not in pattern_to_type:
                pattern_to_type[pattern_tuple] = type_counter
                type_counter += 1

        print("Pattern -> CG Type mapping:")
        for pattern_tuple, cg_type in sorted(pattern_to_type.items(), key=lambda x: x[1]):
            print(f"  {list(pattern_tuple)} -> Type {cg_type} ({len(pattern_tuple)} atoms)")
        print(f"Center atom type: {cg_config.get('center_atom_type', 1)}")

        # Print manual assignments
        cg_assignments = cg_config.get('cg_assignments', [])
        if cg_assignments:
            print("Manual atom assignments:")
            for assignment in cg_assignments:
                atom_ids = assignment.get('atom_ids', [])
                assigned_type = assignment.get('cg_type', 1)
                desc = assignment.get('description', '')
                print(f"  Atom IDs {atom_ids} -> CG Type {assigned_type} ({len(atom_ids)} atoms)" + (f" ({desc})" if desc else ""))
    else:
        print(f"Target atom type: {cg_config.get('target_atom_type', 1)}")
    print(f"Base directory: {paths_config['base_dir']}")
    print(f"Output directory: {paths_config['output_base_dir']}")
    print(f"Export CG trajectory: {cg_config.get('export_cg_trajectory', False)}")

    # Print trajectory filter status
    filter_config = processing_config.get('trajectory_filter', {})
    if filter_config.get('enabled', False):
        print(f"Trajectory filter: ENABLED")
        per_temp = filter_config.get('per_temp', {})
        for sim in config['simulations']:
            sim_name = sim['name']
            if not sim.get('enabled', True):
                continue
            if sim_name in per_temp:
                sim_filter = per_temp[sim_name]
                temps = sim.get('temperatures')
                if temps is not None:
                    # Temperature simulation
                    temp_limits = sim_filter.get('temperatures') or {}
                    if temp_limits:
                        limit_str = ', '.join([f"{t}K:{temp_limits.get(str(t), 'ALL')}" for t in temps])
                        print(f"  {sim_name}: {limit_str}")
                    else:
                        fallback = sim_filter.get('max_trajectories')
                        fallback_str = 'ALL' if fallback is None else fallback
                        print(f"  {sim_name}: {fallback_str}")
                else:
                    # Non-temperature simulation
                    max_traj = sim_filter.get('max_trajectories')
                    max_traj_str = 'ALL' if max_traj is None else max_traj
                    print(f"  {sim_name}: {max_traj_str}")
    else:
        print(f"Trajectory filter: DISABLED")
    print(f"{'='*60}\n")

    # Process each simulation
    all_stats = []
    enabled_sims = [s for s in config['simulations'] if s.get("enabled", True)]

    for sim_config in enabled_sims:
        stats = process_simulation(
            sim_config, paths_config, cg_config, output_config, processing_config
        )
        all_stats.append(stats)

    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_files = 0
    total_filtered_files = 0
    total_success = 0
    total_failed = 0
    total_timesteps = 0
    total_particles = 0

    for stats in all_stats:
        if not stats['enabled']:
            continue

        total_files += stats['total_files']
        total_filtered_files += stats['filtered_files']
        total_success += stats['success']
        total_failed += stats['failed']
        total_timesteps += stats['total_timesteps']
        total_particles += stats['total_particles']

        if stats['total_files'] > 0:
            print(f"\n{stats['name']}:")
            if stats['total_files'] != stats['filtered_files']:
                print(f"  Files scanned: {stats['total_files']}")
                print(f"  Files filtered: {stats['filtered_files']}")
            print(f"  Files processed: {stats['success']}/{stats['filtered_files']}")
            if stats['failed'] > 0:
                print(f"  Failed: {stats['failed']}")
            if stats['total_timesteps'] > 0:
                print(f"  Timesteps: {stats['total_timesteps']:,}")
            if stats['total_particles'] > 0:
                print(f"  Particles: {stats['total_particles']:,}")

    print(f"\n{'='*60}")
    print("TOTAL:")
    if total_files != total_filtered_files:
        print(f"  Files scanned: {total_files}")
        print(f"  Files filtered: {total_filtered_files}")
    print(f"  Files processed: {total_success}/{total_filtered_files}")
    if total_failed > 0:
        print(f"  Failed: {total_failed}")
    if total_timesteps > 0:
        print(f"  Timesteps: {total_timesteps:,}")
    if total_particles > 0:
        print(f"  Particles: {total_particles:,}")
    print(f"{'='*60}\n")

    # Print errors if any
    has_errors = any(len(s['errors']) > 0 for s in all_stats)
    if has_errors:
        print("Errors:")
        for stats in all_stats:
            for err in stats['errors']:
                print(f"  [{stats['name']}] {err}")
        print()

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
