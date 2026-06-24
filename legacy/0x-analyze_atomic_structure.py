#!/usr/bin/env python3
"""
Atomic Structure Analysis Script

Analyzes atomic structures using SOAP descriptors, PCA, t-SNE, and GNN.
Features:
1. Translation and rotation invariant relative coordinates
2. SOAP descriptor analysis
3. PCA and t-SNE dimensionality reduction
4. Graph neural network embeddings
5. Structure classification and visualization
6. Outlier extraction to new CSV files

Author: CH_CG Workflow
Date: 2025-12-31
"""

import os
import json
import argparse
import glob
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial.distance import pdist, squareform

# Using CPU-based sklearn algorithms

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not available. Graph topology visualization will be limited.")

# Try to import optional packages
try:
    from ase.io import read_lammps_dump
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("Warning: ASE not available. Will use custom LAMMPS reader.")

try:
    from dscribe.descriptors import SOAP
    HAS_DSCRIBE = True
    # Test SOAP import
    try:
        test_soap = SOAP(species=[1], rcut=4.0, n_max=4, l_max=4)
    except TypeError:
        HAS_DSCRIBE = False
        print("Warning: dscribe SOAP API incompatible. Will use simplified descriptors.")
except ImportError:
    HAS_DSCRIBE = False
    print("Warning: dscribe not available. Will use simplified descriptors.")

# PyTorch/PyTorch Geometric not required for this analysis
HAS_TORCH = False

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


class LAMMPSTrajectoryReader:
    """Reader for LAMMPS dump files with atomic positions and energies."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)

    def read_frame(self):
        """Read a single frame from LAMMPS dump file."""
        atoms = []
        box = None
        timestep = None
        energies = []
        forces = []

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(lines[i+1].strip())
                i += 2
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms = int(lines[i+1].strip())
                i += 2
            elif line.startswith("ITEM: BOX BOUNDS"):
                i += 1
                box_bounds = []
                for _ in range(3):
                    bounds = lines[i].strip().split()
                    box_bounds.append([float(bounds[0]), float(bounds[1])])
                    i += 1
                box = np.array(box_bounds)
            elif line.startswith("ITEM: ATOMS"):
                i += 1
                columns = line.split()[2:]  # Skip "ITEM: ATOMS"
                atom_data = []
                for _ in range(n_atoms):
                    values = lines[i].strip().split()
                    atom_data.append([float(v) for v in values])
                    i += 1

                atom_data = np.array(atom_data)

                # Parse based on column names
                col_map = {name: idx for idx, name in enumerate(columns)}

                ids = atom_data[:, col_map['id']].astype(int)
                types = atom_data[:, col_map['type']].astype(int)
                positions = atom_data[:, [col_map['xu'], col_map['yu'], col_map['zu']]]

                energies = None
                forces = None
                if 'c_pe' in col_map:
                    energies = atom_data[:, col_map['c_pe']]
                if 'fx' in col_map:
                    forces = atom_data[:, [col_map['fx'], col_map['fy'], col_map['fz']]]

                break
            else:
                i += 1

        return {
            'timestep': timestep,
            'positions': positions,
            'types': types,
            'ids': ids,
            'box': box,
            'energies': energies,
            'forces': forces
        }


class CGTrajectoryReader:
    """Reader for coarse-grained LAMMPS trajectory files."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)

    def read_all_frames(self):
        """Read all frames from CG lammpstrj file."""
        frames = []

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(lines[i+1].strip())
                i += 2
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms = int(lines[i+1].strip())
                i += 2
            elif line.startswith("ITEM: BOX BOUNDS"):
                i += 1
                box_bounds = []
                for _ in range(3):
                    bounds = lines[i].strip().split()
                    box_bounds.append([float(bounds[0]), float(bounds[1])])
                    i += 1
                box = np.array(box_bounds)
            elif line.startswith("ITEM: ATOMS"):
                i += 1
                columns = line.split()[2:]
                atom_data = []
                for _ in range(n_atoms):
                    if i >= len(lines):
                        break
                    values = lines[i].strip().split()
                    if len(values) > 0:
                        atom_data.append([float(v) for v in values])
                    i += 1

                atom_data = np.array(atom_data)
                col_map = {name: idx for idx, name in enumerate(columns)}

                ids = atom_data[:, col_map['id']].astype(int)
                types = atom_data[:, col_map['type']].astype(int)

                # Try different coordinate column names
                if 'xu' in col_map and 'yu' in col_map and 'zu' in col_map:
                    positions = atom_data[:, [col_map['xu'], col_map['yu'], col_map['zu']]]
                elif 'x' in col_map and 'y' in col_map and 'z' in col_map:
                    positions = atom_data[:, [col_map['x'], col_map['y'], col_map['z']]]
                else:
                    # Fall back to first 3 columns after id and type
                    positions = atom_data[:, 2:5]

                energies = None
                forces = None
                if 'c_pe' in col_map:
                    energies = atom_data[:, col_map['c_pe']]
                if 'fx' in col_map:
                    forces = atom_data[:, [col_map['fx'], col_map['fy'], col_map['fz']]]

                frames.append({
                    'timestep': timestep,
                    'positions': positions,
                    'types': types,
                    'ids': ids,
                    'box': box,
                    'energies': energies,
                    'forces': forces
                })
            else:
                i += 1

        return frames


class StructureDescriptor:
    """Compute structure descriptors with translation/rotation invariance."""

    def __init__(self, rcut=5.0, n_max=8, l_max=6):
        self.rcut = rcut
        self.n_max = n_max
        self.l_max = l_max
        self.soap = None

        # Initialize SOAP later when computing (to avoid init errors)
        self._soap_params = {
            'species': [1, 2],  # Carbon and Hydrogen
            'rcut': rcut,
            'n_max': n_max,
            'l_max': l_max,
            'sigma': 0.5,
            'periodic': True,
            'average': 'inner'
        }

    def compute_relative_positions(self, positions, box):
        """
        Convert to relative positions for translation invariance.
        Use center of mass as reference.
        """
        # Center the system
        center = positions.mean(axis=0)
        rel_positions = positions - center

        return rel_positions

    def compute_rotation_invariant_features(self, positions, types, box):
        """
        Compute rotation-invariant features (distances, angles).
        """
        rel_pos = self.compute_relative_positions(positions, box)
        n_atoms = len(positions)

        features = []

        # Pairwise distances (rotation invariant)
        for i in range(n_atoms):
            for j in range(i+1, min(i+100, n_atoms)):  # Limit for efficiency
                dr = rel_pos[i] - rel_pos[j]
                # Apply minimum image convention
                box_lengths = box[:, 1] - box[:, 0]
                dr -= np.round(dr / box_lengths) * box_lengths
                dist = np.linalg.norm(dr)
                features.append([dist, types[i], types[j]])

        return np.array(features)

    def compute_soap_descriptor(self, frame_data):
        """Compute SOAP descriptor for the structure."""
        if not HAS_DSCRIBE:
            # Fallback: use simplified features
            return self.compute_rotation_invariant_features(
                frame_data['positions'],
                frame_data['types'],
                frame_data['box']
            ).flatten()[:500]  # Fixed size

        # Initialize SOAP on first use
        if self.soap is None:
            try:
                # Try different parameter combinations for dscribe compatibility
                try:
                    self.soap = SOAP(**self._soap_params)
                except TypeError:
                    # Older dscribe version might have different params
                    self.soap = SOAP(
                        species=[1, 2],
                        rcut=self.rcut,
                        n_max=self.n_max,
                        l_max=self.l_max
                    )
            except Exception as e:
                print(f"Warning: SOAP initialization failed: {e}. Using fallback features.")
                return self.compute_rotation_invariant_features(
                    frame_data['positions'],
                    frame_data['types'],
                    frame_data['box']
                ).flatten()[:500]

        from ase import Atoms
        # Create ASE Atoms object
        box = frame_data['box']
        cell = [[box[0,1]-box[0,0], 0, 0],
                [0, box[1,1]-box[1,0], 0],
                [0, 0, box[2,1]-box[2,0]]]

        atoms = Atoms(
            positions=frame_data['positions'],
            numbers=frame_data['types'],
            cell=cell,
            pbc=True
        )

        try:
            soap_desc = self.soap.create(atoms)
            return soap_desc.flatten()
        except Exception as e:
            print(f"Warning: SOAP computation failed: {e}. Using fallback features.")
            return self.compute_rotation_invariant_features(
                frame_data['positions'],
                frame_data['types'],
                frame_data['box']
            ).flatten()[:500]


class GraphNeuralNetwork:
    """Simple GNN for structure embedding."""

    def __init__(self, input_dim=64, hidden_dim=64, output_dim=32):
        self.model = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def atoms_to_graph(self, frame_data, feat_dim=16):
        """Convert atomic structure to graph."""
        if not HAS_TORCH:
            return None

        positions = frame_data['positions']
        types = frame_data['types']
        n_atoms = len(positions)

        # Create node features (one-hot encoded atom types + some positional info)
        node_features = []
        for i in range(n_atoms):
            feat = np.zeros(feat_dim)
            feat[types[i]-1] = 1  # Atom type one-hot
            # Add some positional encoding
            for j, pos in enumerate(positions[i]):
                feat[2 + j*3] = np.sin(pos)
                feat[3 + j*3] = np.cos(pos)
            node_features.append(feat)

        node_features = torch.tensor(node_features, dtype=torch.float)

        # Create edges based on distance cutoff
        edge_index = []
        for i in range(n_atoms):
            for j in range(i+1, min(i+20, n_atoms)):
                edge_index.append([i, j])
                edge_index.append([j, i])

        if len(edge_index) == 0:
            return None

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=node_features, edge_index=edge_index)

    def compute_embedding(self, graph):
        """Compute graph embedding."""
        if not HAS_TORCH or graph is None:
            return np.random.randn(32)

        # Adjust model input dimension if needed
        input_dim = graph.x.shape[1]
        if self.model is None or not hasattr(self.model, 'conv1'):
            # Recreate model with correct input dimension
            class GNNModel(nn.Module):
                def __init__(self, in_dim, hid_dim, out_dim):
                    super().__init__()
                    self.conv1 = GCNConv(in_dim, hid_dim)
                    self.conv2 = GCNConv(hid_dim, hid_dim)
                    self.conv3 = GCNConv(hid_dim, out_dim)
                    self.lin = nn.Linear(out_dim, out_dim)

                def forward(self, data):
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    x = F.relu(self.conv1(x, edge_index))
                    x = F.relu(self.conv2(x, edge_index))
                    x = F.relu(self.conv3(x, edge_index))
                    x = global_mean_pool(x, batch)
                    return self.lin(x)

            self.model = GNNModel(input_dim, 64, 32)
            self.model.eval()

        with torch.no_grad():
            graph = graph.to('cpu')
            # Add batch if not present
            if not hasattr(graph, 'batch'):
                graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            embedding = self.model(graph)
        return embedding.numpy().flatten()


class AtomicStructureAnalyzer:
    """Main analyzer for atomic and CG structures."""

    def __init__(self, base_dir=None, config_file=None, mode='cg'):
        """
        Initialize the analyzer.

        Parameters
        ----------
        base_dir : str or Path, optional
            Base directory for data. If None, uses path from config file.
        config_file : str or Path, optional
            Path to config file. If None, uses default config.
        mode : str, default='cg'
            Analysis mode: 'cg' for coarse-grained, 'aa' for all-atom
        """
        self.mode = mode
        self.base_dir = Path(base_dir) if base_dir else None

        # Load config
        if config_file:
            config_path = Path(config_file)
        else:
            # Use mode-specific default config
            if mode == 'cg':
                config_path = Path(__file__).parent / "config.json"
            else:
                config_path = Path(__file__).parent / "config_structure.json"

        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config(mode)

        # Setup base directory from config if not provided
        if self.base_dir is None:
            paths = self.config.get('paths', {})
            if mode == 'cg' and 'cg_data_base_dir' in paths:
                self.base_dir = Path(paths['cg_data_base_dir'])
            elif mode == 'aa' and 'aa_data_base_dir' in paths:
                self.base_dir = Path(paths['aa_data_base_dir'])
            else:
                self.base_dir = Path('/mnt/d/Workbench/CH_CG')

        # Setup output directory - use paths.output_base_dir if available
        if 'paths' in self.config and 'output_base_dir' in self.config['paths']:
            self.output_dir = Path(self.config['paths']['output_base_dir'])
        else:
            self.output_dir = Path(self.config.get('output_base_dir',
                                                   self.base_dir / "structure_analysis_results"))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        # Read SOAP parameters from config
        soap_config = self.config.get('soap', {})
        self.descriptor = StructureDescriptor(
            rcut=soap_config.get('rcut', 5.0),
            n_max=soap_config.get('n_max', 8),
            l_max=soap_config.get('l_max', 6)
        )
        self.gnn = GraphNeuralNetwork()

        # Data storage
        self.structures = []
        self.descriptors = []
        self.embeddings = []
        self.pca_result = None
        self.tsne_result = None
        self.gnn_result = None
        self.labels = None

    def _get_default_config(self, mode='cg'):
        """
        Get default configuration based on mode.

        Parameters
        ----------
        mode : str
            'cg' for coarse-grained, 'aa' for all-atom

        Returns
        -------
        dict
            Default configuration
        """
        if mode == 'cg':
            return {
                'paths': {
                    'cg_data_base_dir': '/mnt/d/Workbench/CH_CG/02.cg_dataset',
                    'output_base_dir': '/mnt/d/Workbench/CH_CG/03-1.cgdata_analysis/results'
                },
                'analysis': {
                    'max_frames': 5000,
                    'max_per_file': 1000,
                    'n_neighbors': 10,
                    'min_samples': 5
                },
                'soap': {'rcut': 5.0, 'n_max': 8, 'l_max': 6},
                'pca': {'n_components': 10},
                'tsne': {'n_components': 2, 'perplexity': 30},
                'clustering': {'method': 'dbscan', 'min_samples': 5, 'n_clusters': 4}
            }
        else:  # aa mode
            return {
                'paths': {
                    'aa_data_base_dir': '/mnt/d/Workbench/CH_CG/01.aa',
                    'output_base_dir': '/mnt/d/Workbench/CH_CG/03-1.cgdata_analysis/structure_analysis_results'
                },
                'analysis': {
                    'max_frames': 500,
                    'n_neighbors': 10,
                    'min_samples': 5,
                    'outlier_threshold': 2.0
                },
                'soap': {'rcut': 5.0, 'n_max': 8, 'l_max': 6},
                'clustering': {'n_clusters': 4}
            }

    def find_trajectory_files(self):
        """Find all LAMMPS trajectory files (AA mode)."""
        # Safely get aa_data_base_dir from config
        paths = self.config.get('paths', {})
        aa_dir = Path(paths.get('aa_data_base_dir', '/mnt/d/Workbench/CH_CG/01.aa'))
        pattern = str(aa_dir / "*/traj/*")
        files = glob.glob(pattern)
        return sorted(files)

    def find_cg_trajectory_files(self):
        """Find all CG trajectory files (*.lammpstrj) in cg_dataset directory."""
        # Try multiple sources for cg_data_base_dir
        cg_base_dir = None
        if 'paths' in self.config:
            cg_base_dir = self.config['paths'].get('cg_data_base_dir')
        if cg_base_dir is None:
            cg_base_dir = self.config.get('cg_data_base_dir')
        if cg_base_dir is None:
            cg_base_dir = '/mnt/d/Workbench/CH_CG/02.cg_dataset'

        cg_base_dir = Path(cg_base_dir)
        # Pattern matches: sim_type/temperature/*_cg.lammpstrj (e.g., 1-npt/200/NPT.200.xxx_cg.lammpstrj)
        pattern = str(cg_base_dir / "*/*/*_cg.lammpstrj")
        files = glob.glob(pattern)
        return sorted(files)

    def load_cg_trajectories(self, max_frames=None, max_per_file=None):
        """Load CG trajectory data from files.

        Returns
        -------
        cg_data : list of dict
            List of CG frame data with metadata
        """
        files = self.find_cg_trajectory_files()

        # Filter for specified simulation types only (1-npt, 2-nvt, 3-upT, 4-dnT)
        allowed_sim_types = self.config.get('simulations', [])
        allowed_names = set()
        for sim in allowed_sim_types:
            if sim.get('enabled', True):
                allowed_names.add(sim['name'])

        if allowed_names:
            filtered_files = []
            for f in files:
                parts = Path(f).parts
                sim_type = parts[-3] if len(parts) >= 3 else 'unknown'
                if sim_type in allowed_names:
                    filtered_files.append(f)
            files = filtered_files
            print(f"Found {len(files)} CG trajectory files (filtered by allowed types: {sorted(allowed_names)})")
        else:
            print(f"Found {len(files)} CG trajectory files")

        if max_frames is None:
            # Try data_loading config first (for config_cg_particle.json), then fall back to analysis config
            max_frames = self.config.get('data_loading', {}).get('max_frames',
                        self.config['analysis'].get('max_frames', 500))

        if max_per_file is None:
            # Try data_loading config first, then fall back to analysis config
            max_per_file = self.config.get('data_loading', {}).get('max_per_file',
                          self.config['analysis'].get('max_per_file', 10))

        cg_data = []
        frame_count = 0

        for filepath in files:
            if max_frames is not None and frame_count >= max_frames:
                break

            # Extract metadata from filename
            # Expected path: .../02.cg_dataset/1-npt/200/NPT.200.*_cg.lammpstrj
            parts = Path(filepath).parts
            sim_type = parts[-3] if len(parts) >= 3 else 'unknown'

            # Extract temperature from directory name
            try:
                temp = int(parts[-2])
            except (ValueError, IndexError):
                temp = 300  # Default temperature

            # Find corresponding _particles.csv file for energy data
            basename = Path(filepath).stem.replace('_cg', '')
            particles_csv = Path(filepath).parent / f"{basename}_particles.csv"

            reader = CGTrajectoryReader(filepath)
            try:
                frames = reader.read_all_frames()

                # Read energy from _particles.csv if available
                total_energy = None
                if particles_csv.exists():
                    try:
                        import pandas as pd
                        df = pd.read_csv(particles_csv)
                        if 'c_pe' in df.columns:
                            total_energy = df['c_pe'].sum()
                    except Exception as e:
                        pass  # Fall back to trajectory energy

                for frame in frames:
                    if max_frames is not None and frame_count >= max_frames:
                        break
                    if max_per_file is not None and len([d for d in cg_data if d['source_file'] == filepath]) >= max_per_file:
                        break

                    frame['source_file'] = filepath
                    frame['sim_type'] = sim_type
                    frame['temperature'] = temp
                    # Use total energy from CSV if available
                    if total_energy is not None:
                        # Store as a scalar value
                        frame['total_energy'] = total_energy
                        # Also update the energies array for backward compatibility
                        if frame['energies'] is not None:
                            frame['energies'] = np.array([total_energy])
                        else:
                            frame['energies'] = np.array([total_energy])
                    cg_data.append(frame)
                    frame_count += 1

            except Exception as e:
                print(f"  Warning: Failed to load {filepath}: {e}")

        print(f"Successfully loaded {len(cg_data)} CG frames from {len(files)} files")
        self.cg_structures = cg_data
        return cg_data

    def compute_cg_descriptors(self, cg_data=None):
        """Compute descriptors for CG structures."""
        if cg_data is None:
            cg_data = self.cg_structures

        if cg_data is None or len(cg_data) == 0:
            print("No CG structures available")
            return None

        print("Computing CG structure descriptors...")

        descriptors = []
        for i, struct in enumerate(cg_data):
            desc = self.descriptor.compute_soap_descriptor(struct)
            descriptors.append(desc)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(cg_data)} structures...")

        self.cg_descriptors = np.array(descriptors)
        print(f"CG descriptor shape: {self.cg_descriptors.shape}")

        # Normalize descriptors
        scaler = StandardScaler()
        self.cg_descriptors = scaler.fit_transform(self.cg_descriptors)

        return self.cg_descriptors

    def load_trajectories(self, max_frames=None):
        """Load trajectory data from files."""
        files = self.find_trajectory_files()
        print(f"Found {len(files)} trajectory files")

        if max_frames is None:
            # Try data_loading config first (for config_cg_particle.json), then fall back to analysis config
            max_frames = self.config.get('data_loading', {}).get('max_frames',
                        self.config['analysis'].get('max_frames', 500))

        structures = []
        frame_count = 0

        for filepath in files:
            if frame_count >= max_frames:
                break

            reader = LAMMPSTrajectoryReader(filepath)
            try:
                frame = reader.read_frame()
                frame['source_file'] = filepath
                frame['sim_type'] = filepath.split('/')[-3]  # Extract sim type
                structures.append(frame)
                frame_count += 1

                if frame_count % 50 == 0:
                    print(f"  Loaded {frame_count} frames...")
            except Exception as e:
                print(f"  Warning: Failed to load {filepath}: {e}")

        print(f"Successfully loaded {len(structures)} frames")
        self.structures = structures
        return structures

    def compute_descriptors(self):
        """Compute structure descriptors for all frames."""
        print("Computing structure descriptors...")

        descriptors = []
        for i, struct in enumerate(self.structures):
            desc = self.descriptor.compute_soap_descriptor(struct)
            descriptors.append(desc)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(self.structures)} structures...")

        self.descriptors = np.array(descriptors)
        print(f"Descriptor shape: {self.descriptors.shape}")

        # Normalize descriptors
        scaler = StandardScaler()
        self.descriptors = scaler.fit_transform(self.descriptors)

        return self.descriptors

    def compute_gnn_embeddings(self):
        """Compute GNN embeddings for all frames."""
        if not HAS_TORCH:
            print("GNN not available, using random embeddings")
            self.gnn_result = np.random.randn(len(self.structures), 32)
            return self.gnn_result

        print("Computing GNN embeddings...")

        embeddings = []
        for i, struct in enumerate(self.structures):
            graph = self.gnn.atoms_to_graph(struct)
            emb = self.gnn.compute_embedding(graph)
            embeddings.append(emb)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(self.structures)} structures...")

        self.gnn_result = np.array(embeddings)
        print(f"GNN embedding shape: {self.gnn_result.shape}")

        return self.gnn_result

    def perform_pca(self, n_components=3):
        """
        Perform PCA dimensionality reduction.

        Parameters
        ----------
        n_components : int
            Number of principal components

        Returns
        -------
        pca_result : ndarray
            PCA transformed data
        pca : object
            Fitted PCA model
        """
        print(f"Performing PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(self.descriptors)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

        return self.pca_result, pca

    def perform_tsne(self, n_components=2, perplexity=30):
        """
        Perform t-SNE dimensionality reduction.

        Parameters
        ----------
        n_components : int
            Number of t-SNE components
        perplexity : float
            t-SNE perplexity parameter

        Returns
        -------
        tsne_result : ndarray
            t-SNE transformed data
        """
        print(f"Performing t-SNE (n_components={n_components}, perplexity={perplexity})...")
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,
            verbose=1
        )
        self.tsne_result = tsne.fit_transform(self.descriptors)
        print(f"t-SNE result shape: {self.tsne_result.shape}")

        return self.tsne_result

    def cluster_structures(self, method='kmeans'):
        """
        Cluster structures based on descriptors.

        Parameters
        ----------
        method : str
            Clustering method ('kmeans' or 'dbscan')

        Returns
        -------
        labels : ndarray
            Cluster labels for each structure
        """
        n_neighbors = self.config['analysis'].get('n_neighbors', 10)
        min_samples = self.config['analysis'].get('min_samples', 5)
        n_clusters = self.config.get('clustering', {}).get('n_clusters', 4)

        if method == 'dbscan':
            print(f"Clustering using DBSCAN...")
            from scipy.spatial.distance import pdist
            distances = pdist(self.pca_result[:, :3])
            eps = np.percentile(distances, 30)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            print(f"Clustering using K-Means...")
            # Use K-Means with number of clusters based on data size
            n_clust = min(n_clusters, len(self.pca_result) // 2)
            clusterer = KMeans(n_clusters=n_clust, random_state=42)

        # Use PCA result for clustering
        self.labels = clusterer.fit_predict(self.pca_result[:, :3])

        n_clusters_found = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        print(f"Found {n_clusters_found} clusters")

        return self.labels

    def plot_pca_analysis(self, use_cg_data=False):
        """
        Plot PCA results with new visualization format.

        Two types of plots:
        1. Overall analysis: left=energy, right=sim_type
        2. Per-sim-type analysis: left=energy, right=temperature (RdBu, 10 levels)
        """
        # Determine which data to use
        if use_cg_data and hasattr(self, 'cg_pca_result') and self.cg_pca_result is not None:
            pca_result = self.cg_pca_result
            structures = self.cg_structures if hasattr(self, 'cg_structures') else []
            prefix = 'CG_'
            pca_obj = self._cg_pca_object if hasattr(self, '_cg_pca_object') else None
        else:
            pca_result = self.pca_result
            structures = self.structures
            prefix = ''
            pca_obj = self._pca_object if hasattr(self, '_pca_object') else None

        if pca_result is None:
            print("No PCA results available")
            return

        # Check if structures have temperature field
        has_temp = len(structures) > 0 and 'temperature' in structures[0]

        # Extract metadata
        energies = []
        sim_types = []
        temperatures = []

        for struct in structures:
            # Check for total_energy first (from CSV), then fallback to energies
            if 'total_energy' in struct:
                energies.append(struct['total_energy'])
            elif struct['energies'] is not None and len(struct['energies']) > 0:
                energies.append(struct['energies'].sum() if hasattr(struct['energies'], 'sum') else struct['energies'][0])
            else:
                energies.append(0)
            sim_types.append(struct.get('sim_type', 'unknown'))
            temperatures.append(struct.get('temperature', 300))

        energies = np.array(energies)
        sim_types = np.array(sim_types)
        temperatures = np.array(temperatures)

        # Get unique sim types
        unique_sim_types = sorted(list(set(sim_types)))

        # Helper function to get variance
        def get_var(i):
            if pca_obj is not None and i < len(pca_obj.explained_variance_ratio_):
                return 100 * pca_obj.explained_variance_ratio_[i]
            return 0

        # === Overall Analysis ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Energy-colored
        ax = axes[0]
        norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                            c=energies, cmap='viridis', norm=norm, alpha=0.6, s=30)
        ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
        ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
        ax.set_title('Overall: PC1 vs PC2 (colored by energy)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Energy')

        # Right: Sim type-colored
        ax = axes[1]
        colors_sim = plt.cm.Set3(np.linspace(0, 1, len(unique_sim_types)))
        color_map = {t: colors_sim[i] for i, t in enumerate(unique_sim_types)}

        for sim_type in unique_sim_types:
            mask = sim_types == sim_type
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                      c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)

        ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
        ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
        ax.set_title('Overall: PC1 vs PC2 (colored by sim type)')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.suptitle(f'PCA Analysis - Overall ({prefix}CG Data)' if use_cg_data else 'PCA Analysis - Overall',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{prefix}pca_overall.png', bbox_inches='tight', dpi=300)
        print(f"Saved: {self.output_dir / 'figures' / f'{prefix}pca_overall.png'}")
        plt.close()

        # === Per-Sim-Type Analysis ===
        if has_temp:
            unique_temps = sorted(list(set(temperatures)))
            n_temps = len(unique_temps)

            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                if np.sum(mask) == 0:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                pca_subset = pca_result[mask]
                energies_subset = energies[mask]
                temps_subset = temperatures[mask]

                # Left: Energy-colored
                ax = axes[0]
                norm = plt.Normalize(vmin=energies_subset.min(), vmax=energies_subset.max())
                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                    c=energies_subset, cmap='viridis', norm=norm, alpha=0.6, s=30)
                ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
                ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
                ax.set_title(f'{sim_type}: PC1 vs PC2 (colored by energy)')
                ax.grid(alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Energy')

                # Right: Temperature-colored (RdBu, 10 levels)
                ax = axes[1]
                # Create 10 discrete levels for temperature
                from matplotlib.colors import BoundaryNorm
                bounds = np.linspace(min(unique_temps) - 25, max(unique_temps) + 25, 11)
                temp_norm_disc = BoundaryNorm(bounds, ncolors=10)

                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                    c=temps_subset, cmap=plt.cm.RdBu, norm=temp_norm_disc, alpha=0.6, s=30)
                ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
                ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
                ax.set_title(f'{sim_type}: PC1 vs PC2 (colored by temperature)')
                ax.grid(alpha=0.3)
                cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                # Set tick labels to show actual temperatures
                if len(unique_temps) <= 10:
                    cbar.set_ticks(unique_temps)

                plt.suptitle(f'PCA Analysis - {sim_type} ({prefix}CG Data)' if use_cg_data else f'PCA Analysis - {sim_type}',
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                safe_sim_type = sim_type.replace('-', '_').replace('/', '_')
                plt.savefig(self.output_dir / 'figures' / f'{prefix}pca_{safe_sim_type}.png', bbox_inches='tight', dpi=300)
                print(f"Saved: {self.output_dir / 'figures' / f'{prefix}pca_{safe_sim_type}.png'}")
                plt.close()

    def _get_pca_variances(self):
        """Get PCA variance values."""
        if hasattr(self, '_pca_object'):
            return self._pca_object.explained_variance_ratio_
        return np.ones(10) / 10  # Default

    def _get_pca_variance(self, i):
        """Get single PCA variance."""
        return self._get_pca_variances()[i] if i < len(self._get_pca_variances()) else 0

    def plot_tsne_analysis(self, use_cg_data=False):
        """
        Plot t-SNE results with new visualization format.

        Two types of plots:
        1. Overall analysis: left=energy, right=sim_type
        2. Per-sim-type analysis: left=energy, right=temperature (RdBu, 10 levels)
        """
        # Determine which data to use
        if use_cg_data and hasattr(self, 'cg_tsne_result') and self.cg_tsne_result is not None:
            tsne_result = self.cg_tsne_result
            structures = self.cg_structures if hasattr(self, 'cg_structures') else []
            prefix = 'CG_'
        else:
            tsne_result = self.tsne_result
            structures = self.structures
            prefix = ''

        if tsne_result is None:
            print("No t-SNE results available")
            return

        # Check if structures have temperature field
        has_temp = len(structures) > 0 and 'temperature' in structures[0]

        # Extract metadata
        energies = []
        sim_types = []
        temperatures = []

        for struct in structures:
            # Check for total_energy first (from CSV), then fallback to energies
            if 'total_energy' in struct:
                energies.append(struct['total_energy'])
            elif struct['energies'] is not None and len(struct['energies']) > 0:
                energies.append(struct['energies'].sum() if hasattr(struct['energies'], 'sum') else struct['energies'][0])
            else:
                energies.append(0)
            sim_types.append(struct.get('sim_type', 'unknown'))
            temperatures.append(struct.get('temperature', 300))

        energies = np.array(energies)
        sim_types = np.array(sim_types)
        temperatures = np.array(temperatures)

        # Get unique sim types
        unique_sim_types = sorted(list(set(sim_types)))

        # === Overall Analysis ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Energy-colored
        ax = axes[0]
        norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                            c=energies, cmap='viridis', norm=norm, alpha=0.6, s=30)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('Overall: t-SNE (colored by energy)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Energy')

        # Right: Sim type-colored
        ax = axes[1]
        colors_sim = plt.cm.Set3(np.linspace(0, 1, len(unique_sim_types)))
        color_map = {t: colors_sim[i] for i, t in enumerate(unique_sim_types)}

        for sim_type in unique_sim_types:
            mask = sim_types == sim_type
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                      c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)

        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('Overall: t-SNE (colored by sim type)')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.suptitle(f't-SNE Analysis - Overall ({prefix}CG Data)' if use_cg_data else 't-SNE Analysis - Overall',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{prefix}tsne_overall.png', bbox_inches='tight', dpi=300)
        print(f"Saved: {self.output_dir / 'figures' / f'{prefix}tsne_overall.png'}")
        plt.close()

        # === Per-Sim-Type Analysis ===
        if has_temp:
            unique_temps = sorted(list(set(temperatures)))

            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                if np.sum(mask) == 0:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                tsne_subset = tsne_result[mask]
                energies_subset = energies[mask]
                temps_subset = temperatures[mask]

                # Left: Energy-colored
                ax = axes[0]
                norm = plt.Normalize(vmin=energies_subset.min(), vmax=energies_subset.max())
                scatter = ax.scatter(tsne_subset[:, 0], tsne_subset[:, 1],
                                    c=energies_subset, cmap='viridis', norm=norm, alpha=0.6, s=30)
                ax.set_xlabel('t-SNE Component 1')
                ax.set_ylabel('t-SNE Component 2')
                ax.set_title(f'{sim_type}: t-SNE (colored by energy)')
                ax.grid(alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Energy')

                # Right: Temperature-colored (RdBu, 10 levels)
                ax = axes[1]
                from matplotlib.colors import BoundaryNorm
                bounds = np.linspace(min(unique_temps) - 25, max(unique_temps) + 25, 11)
                temp_norm_disc = BoundaryNorm(bounds, ncolors=10)

                scatter = ax.scatter(tsne_subset[:, 0], tsne_subset[:, 1],
                                    c=temps_subset, cmap=plt.cm.RdBu, norm=temp_norm_disc, alpha=0.6, s=30)
                ax.set_xlabel('t-SNE Component 1')
                ax.set_ylabel('t-SNE Component 2')
                ax.set_title(f'{sim_type}: t-SNE (colored by temperature)')
                ax.grid(alpha=0.3)
                cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                if len(unique_temps) <= 10:
                    cbar.set_ticks(unique_temps)

                plt.suptitle(f't-SNE Analysis - {sim_type} ({prefix}CG Data)' if use_cg_data else f't-SNE Analysis - {sim_type}',
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                safe_sim_type = sim_type.replace('-', '_').replace('/', '_')
                plt.savefig(self.output_dir / 'figures' / f'{prefix}tsne_{safe_sim_type}.png', bbox_inches='tight', dpi=300)
                print(f"Saved: {self.output_dir / 'figures' / f'{prefix}tsne_{safe_sim_type}.png'}")
                plt.close()

    def plot_gnn_analysis(self):
        """Plot GNN embedding results."""
        if self.gnn_result is None or self.gnn_result.shape[1] < 2:
            print("No GNN results available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Get energies for coloring
        energies = []
        sim_types = []
        for struct in self.structures:
            if struct['energies'] is not None:
                energies.append(struct['energies'].mean())
            else:
                energies.append(0)
            sim_types.append(struct.get('sim_type', 'unknown'))

        energies = np.array(energies)
        norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
        cmap = plt.cm.viridis

        # Left: GNN embedding colored by energy
        ax = axes[0]
        scatter = ax.scatter(self.gnn_result[:, 0], self.gnn_result[:, 1],
                            c=energies, cmap=cmap, norm=norm, alpha=0.6, s=30)
        ax.set_xlabel('GNN Dimension 1')
        ax.set_ylabel('GNN Dimension 2')
        ax.set_title('GNN Embedding: Colored by Energy')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Energy')

        # Right: GNN embedding colored by simulation type
        ax = axes[1]
        unique_types = list(set(sim_types))
        colors_type = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        color_map = {t: colors_type[i] for i, t in enumerate(unique_types)}

        for sim_type in unique_types:
            mask = np.array(sim_types) == sim_type
            ax.scatter(self.gnn_result[mask, 0], self.gnn_result[mask, 1],
                      c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)

        ax.set_xlabel('GNN Dimension 1')
        ax.set_ylabel('GNN Dimension 2')
        ax.set_title('GNN Embedding: Colored by Simulation Type')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.suptitle('GNN Analysis of Atomic Structures', fontsize=14, fontweight='bold')
        plt.savefig(self.output_dir / 'figures' / 'gnn_analysis.png', bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'figures' / 'gnn_analysis.png'}")
        plt.close()

    def plot_gnn_graph_structure(self, num_examples=6, edge_cutoff=None):
        """
        Visualize the actual GNN graph structure for selected structures.

        Parameters
        ----------
        num_examples : int
            Number of example structures to visualize
        edge_cutoff : float, optional
            Distance cutoff for edges. If None, uses the GNN's internal cutoff.
        """
        if not HAS_TORCH:
            print("GNN graph visualization requires PyTorch/PyG")
            return

        print(f"Visualizing GNN graph structure for {num_examples} examples...")

        # Select diverse examples (based on PCA space)
        if self.pca_result is not None:
            # Pick structures from different regions of PCA space
            indices = np.linspace(0, len(self.structures)-1, num_examples, dtype=int)
        else:
            indices = range(min(num_examples, len(self.structures)))

        n_cols = 3
        n_rows = (num_examples + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(18, 6*n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

        for plot_idx, struct_idx in enumerate(indices):
            if plot_idx >= num_examples:
                break

            struct = self.structures[struct_idx]
            row = plot_idx // n_cols
            col = plot_idx % n_cols

            # Build graph for this structure
            graph = self.gnn.atoms_to_graph(struct)
            if graph is None:
                continue

            positions = struct['positions']
            types = struct['types']
            n_atoms = len(positions)

            # Determine edge cutoff
            if edge_cutoff is None:
                box = struct['box']
                box_lengths = box[:, 1] - box[:, 0]
                edge_cutoff = min(box_lengths) * 0.3  # 30% of smallest box dimension

            # Rebuild edges with specified cutoff for visualization
            edges = []
            edge_distances = []
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    dr = positions[i] - positions[j]
                    # Minimum image convention
                    box_lengths = struct['box'][:, 1] - struct['box'][:, 0]
                    dr -= np.round(dr / box_lengths) * box_lengths
                    dist = np.linalg.norm(dr)
                    if dist < edge_cutoff:
                        edges.append((i, j))
                        edge_distances.append(dist)

            # Create 3D subplot
            ax = fig.add_subplot(gs[row, col], projection='3d')

            # Plot atoms as scatter points
            colors = ['#1f77b4' if t == 1 else '#ff7f0e' for t in types]  # C=blue, H=orange
            sizes = [100 if t == 1 else 50 for t in types]  # C larger, H smaller
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidth=0.5)

            # Plot edges
            for (i, j), dist in zip(edges, edge_distances):
                pos_i = positions[i]
                pos_j = positions[j]
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], [pos_i[2], pos_j[2]],
                       'gray', alpha=0.3, linewidth=0.5)

            # Get energy for title
            energy = struct['energies'].mean() if struct['energies'] is not None else 'N/A'
            sim_type = struct.get('sim_type', 'unknown')

            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title(f'#{struct_idx} | {sim_type}\n'
                        f'Atoms: {n_atoms} | Edges: {len(edges)}\n'
                        f'E: {energy:.2f} eV' if isinstance(energy, float) else f'#{struct_idx} | {sim_type}\nAtoms: {n_atoms} | Edges: {len(edges)}',
                        fontsize=10)

            # Set equal aspect ratio
            max_range = np.array([positions[:, i].max() - positions[:, i].min()
                                 for i in range(3)]).max() / 2.0
            mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
            mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
            mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.suptitle(f'GNN Graph Structure Visualization (edge cutoff = {edge_cutoff:.2f} Å)',
                    fontsize=14, fontweight='bold')
        plt.savefig(self.output_dir / 'figures' / 'gnn_graph_structure.png', bbox_inches='tight', dpi=300)
        print(f"Saved: {self.output_dir / 'figures' / 'gnn_graph_structure.png'}")
        plt.close()

        # Additional 2D network visualization if networkx is available
        if HAS_NETWORKX:
            self.plot_gnn_network_topology(indices[:min(4, len(indices))])

    def plot_gnn_network_topology(self, indices):
        """Plot 2D network topology for selected structures."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for ax_idx, struct_idx in enumerate(indices):
            if ax_idx >= 4:
                break

            ax = axes[ax_idx]
            struct = self.structures[struct_idx]

            # Create networkx graph
            G = nx.Graph()
            positions = struct['positions']
            types = struct['types']
            n_atoms = len(positions)

            # Add nodes
            for i in range(n_atoms):
                G.add_node(i, atom_type=types[i],
                          pos=(positions[i, 0], positions[i, 1], positions[i, 2]))

            # Add edges (using same logic as GNN)
            box_lengths = struct['box'][:, 1] - struct['box'][:, 0]
            edge_cutoff = min(box_lengths) * 0.3

            for i in range(n_atoms):
                for j in range(i+1, min(i+20, n_atoms)):  # Same limit as GNN
                    dr = positions[i] - positions[j]
                    dr -= np.round(dr / box_lengths) * box_lengths
                    dist = np.linalg.norm(dr)
                    if dist < edge_cutoff:
                        G.add_edge(i, j, weight=dist)

            # Use spring layout for 2D visualization
            pos_2d = nx.spring_layout(G, seed=42, k=1/np.sqrt(n_atoms))

            # Draw nodes
            node_colors = ['#1f77b4' if G.nodes[i]['atom_type'] == 1 else '#ff7f0e'
                          for i in G.nodes()]
            node_sizes = [200 if G.nodes[i]['atom_type'] == 1 else 100 for i in G.nodes()]

            nx.draw_networkx_nodes(G, pos_2d, ax=ax, node_color=node_colors,
                                  node_size=node_sizes, alpha=0.8, edgecolors='black')

            # Draw edges
            nx.draw_networkx_edges(G, pos_2d, ax=ax, alpha=0.3, width=0.5)

            # Draw labels for a subset of nodes
            if n_atoms <= 20:
                nx.draw_networkx_labels(G, pos_2d, ax=ax, font_size=8)

            # Graph statistics
            degrees = dict(G.degree())
            avg_degree = np.mean(list(degrees.values()))
            clustering = nx.average_clustering(G)

            ax.set_title(f'#{struct_idx} | {struct.get("sim_type", "unknown")}\n'
                        f'Nodes: {n_atoms}, Edges: {G.number_of_edges()}\n'
                        f'Avg Degree: {avg_degree:.2f}, Clustering: {clustering:.3f}')
            ax.axis('off')

        plt.suptitle('GNN Graph Topology (2D Spring Layout)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'gnn_network_topology.png', bbox_inches='tight', dpi=300)
        print(f"Saved: {self.output_dir / 'figures' / 'gnn_network_topology.png'}")
        plt.close()

    def plot_gnn_node_features(self, num_examples=50):
        """Visualize node feature distributions across structures."""
        if not HAS_TORCH or len(self.structures) == 0:
            print("Cannot visualize node features")
            return

        print("Analyzing GNN node features...")

        # Collect node features from multiple structures
        all_features = []
        all_types = []
        struct_indices = []

        n_sample = min(num_examples, len(self.structures))
        indices = np.linspace(0, len(self.structures)-1, n_sample, dtype=int)

        for idx in indices:
            struct = self.structures[idx]
            graph = self.gnn.atoms_to_graph(struct)
            if graph is not None:
                all_features.append(graph.x.numpy())
                all_types.extend(struct['types'])
                struct_indices.extend([idx] * len(struct['types']))

        if len(all_features) == 0:
            print("No valid graphs found")
            return

        all_features = np.vstack(all_features)
        all_types = np.array(all_types)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Feature distribution by atom type
        ax = axes[0, 0]
        for atom_type in [1, 2]:
            mask = all_types == atom_type
            features_subset = all_features[mask]
            feature_means = features_subset.mean(axis=0)
            ax.plot(feature_means, label=f'Type {atom_type} ({"C" if atom_type == 1 else "H"})', marker='o')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean Feature Value')
        ax.set_title('Node Feature Profiles by Atom Type')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Feature heatmap for a subset
        ax = axes[0, 1]
        n_show = min(50, all_features.shape[0])
        im = ax.imshow(all_features[:n_show].T, aspect='auto', cmap='viridis')
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Feature Index')
        ax.set_title('Node Feature Heatmap')
        plt.colorbar(im, ax=ax, label='Feature Value')

        # 3. Feature variance
        ax = axes[1, 0]
        feature_var = all_features.var(axis=0)
        ax.bar(range(len(feature_var)), feature_var, alpha=0.7)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Variance')
        ax.set_title('Feature Variance Across All Nodes')
        ax.grid(alpha=0.3)

        # 4. Feature correlation matrix
        ax = axes[1, 1]
        # Sample features for correlation to avoid overcrowding
        n_feat_sample = min(16, all_features.shape[1])
        corr = np.corrcoef(all_features[:, :n_feat_sample].T)
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        ax.set_title(f'Feature Correlation Matrix (first {n_feat_sample} features)')
        plt.colorbar(im, ax=ax, label='Correlation')

        plt.suptitle('GNN Node Feature Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'gnn_node_features.png', bbox_inches='tight', dpi=300)
        print(f"Saved: {self.output_dir / 'figures' / 'gnn_node_features.png'}")
        plt.close()

    def plot_cluster_analysis(self, use_cg_data=False):
        """
        Plot clustering results with new visualization format.

        Two types of plots:
        1. Overall analysis: left=energy, right=sim_type
        2. Per-sim-type analysis: left=energy, right=temperature (RdBu, 10 levels)
        """
        # Determine which data and labels to use
        if use_cg_data and hasattr(self, 'cg_labels') and self.cg_labels is not None:
            labels = self.cg_labels
            pca_result = self.cg_pca_result if hasattr(self, 'cg_pca_result') else None
            structures = self.cg_structures if hasattr(self, 'cg_structures') else []
            prefix = 'CG_'
        else:
            labels = self.labels
            pca_result = self.pca_result
            structures = self.structures
            prefix = ''

        if labels is None:
            print("No clustering results available")
            return

        if pca_result is None:
            print("No PCA results for cluster visualization")
            return

        # Check if structures have temperature field
        has_temp = len(structures) > 0 and 'temperature' in structures[0]

        # Extract metadata
        energies = []
        sim_types = []
        temperatures = []

        for struct in structures:
            # Check for total_energy first (from CSV), then fallback to energies
            if 'total_energy' in struct:
                energies.append(struct['total_energy'])
            elif struct['energies'] is not None and len(struct['energies']) > 0:
                energies.append(struct['energies'].sum() if hasattr(struct['energies'], 'sum') else struct['energies'][0])
            else:
                energies.append(0)
            sim_types.append(struct.get('sim_type', 'unknown'))
            temperatures.append(struct.get('temperature', 300))

        energies = np.array(energies)
        sim_types = np.array(sim_types)
        temperatures = np.array(temperatures)

        # Get unique sim types
        unique_sim_types = sorted(list(set(sim_types)))

        # === Overall Analysis ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Cluster labels colored by energy
        ax = axes[0]
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                            c=energies, cmap='viridis', alpha=0.6, s=30)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Overall: Clusters (colored by energy)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Energy')

        # Overlay cluster boundaries
        unique_labels = sorted(list(set(labels)))
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                centroid = pca_result[mask].mean(axis=0)
                ax.text(centroid[0], centroid[1], f'C{label}' if label >= 0 else 'Out',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Right: Cluster labels colored by sim type
        ax = axes[1]
        colors_sim = plt.cm.Set3(np.linspace(0, 1, len(unique_sim_types)))
        color_map = {t: colors_sim[i] for i, t in enumerate(unique_sim_types)}

        # First plot by sim type
        for sim_type in unique_sim_types:
            mask = sim_types == sim_type
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                      c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Overall: Clusters (colored by sim type)')
        ax.legend()
        ax.grid(alpha=0.3)

        # Overlay cluster labels
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                centroid = pca_result[mask].mean(axis=0)
                ax.text(centroid[0], centroid[1], f'C{label}' if label >= 0 else 'Out',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.suptitle(f'Cluster Analysis - Overall ({prefix}CG Data)' if use_cg_data else 'Cluster Analysis - Overall',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{prefix}cluster_overall.png', bbox_inches='tight', dpi=300)
        print(f"Saved: {self.output_dir / 'figures' / f'{prefix}cluster_overall.png'}")
        plt.close()

        # === Per-Sim-Type Analysis ===
        if has_temp:
            unique_temps = sorted(list(set(temperatures)))

            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                if np.sum(mask) == 0:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                pca_subset = pca_result[mask]
                energies_subset = energies[mask]
                temps_subset = temperatures[mask]
                labels_subset = labels[mask]

                # Left: Energy-colored
                ax = axes[0]
                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                    c=energies_subset, cmap='viridis', alpha=0.6, s=30)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title(f'{sim_type}: Clusters (colored by energy)')
                ax.grid(alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Energy')

                # Overlay cluster labels for this sim type
                unique_labels_subset = sorted(list(set(labels_subset)))
                for label in unique_labels_subset:
                    mask_l = labels_subset == label
                    if np.sum(mask_l) > 0:
                        centroid = pca_subset[mask_l].mean(axis=0)
                        ax.text(centroid[0], centroid[1], f'C{label}' if label >= 0 else 'Out',
                               fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # Right: Temperature-colored (RdBu, 10 levels)
                ax = axes[1]
                from matplotlib.colors import BoundaryNorm
                bounds = np.linspace(min(unique_temps) - 25, max(unique_temps) + 25, 11)
                temp_norm_disc = BoundaryNorm(bounds, ncolors=10)

                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                    c=temps_subset, cmap=plt.cm.RdBu, norm=temp_norm_disc, alpha=0.6, s=30)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title(f'{sim_type}: Clusters (colored by temperature)')
                ax.grid(alpha=0.3)
                cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                if len(unique_temps) <= 10:
                    cbar.set_ticks(unique_temps)

                # Overlay cluster labels
                for label in unique_labels_subset:
                    mask_l = labels_subset == label
                    if np.sum(mask_l) > 0:
                        centroid = pca_subset[mask_l].mean(axis=0)
                        ax.text(centroid[0], centroid[1], f'C{label}' if label >= 0 else 'Out',
                               fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                plt.suptitle(f'Cluster Analysis - {sim_type} ({prefix}CG Data)' if use_cg_data else f'Cluster Analysis - {sim_type}',
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                safe_sim_type = sim_type.replace('-', '_').replace('/', '_')
                plt.savefig(self.output_dir / 'figures' / f'{prefix}cluster_{safe_sim_type}.png', bbox_inches='tight', dpi=300)
                filename = f'{prefix}cluster_{safe_sim_type}.png'
                print(f"Saved: {self.output_dir / 'figures' / filename}")
                plt.close()

    def plot_combined_analysis(self, use_cg_data=False):
        """
        Plot combined analysis with PCA, t-SNE, and Clustering in one figure.

        Creates a 3x2 grid:
        - Row 1: PCA (left=energy, right=sim_type/temperature)
        - Row 2: t-SNE (left=energy, right=sim_type/temperature)
        - Row 3: Clustering (left=energy, right=sim_type/temperature)

        For overall analysis: right column shows sim_type
        For per-sim-type analysis: right column shows temperature (RdBu, 10 levels)
        """
        # Determine which data to use
        if use_cg_data and hasattr(self, 'cg_pca_result') and self.cg_pca_result is not None:
            pca_result = self.cg_pca_result
            tsne_result = self.cg_tsne_result if hasattr(self, 'cg_tsne_result') else None
            labels = self.cg_labels if hasattr(self, 'cg_labels') else None
            structures = self.cg_structures if hasattr(self, 'cg_structures') else []
            prefix = 'CG_'
            pca_obj = self._cg_pca_object if hasattr(self, '_cg_pca_object') else None
        else:
            pca_result = self.pca_result
            tsne_result = self.tsne_result
            labels = self.labels
            structures = self.structures
            prefix = ''
            pca_obj = self._pca_object if hasattr(self, '_pca_object') else None

        if pca_result is None:
            print("No PCA results available for combined analysis")
            return

        # Check if structures have temperature field
        has_temp = len(structures) > 0 and 'temperature' in structures[0]

        # Extract metadata
        energies = []
        sim_types = []
        temperatures = []

        for struct in structures:
            # Check for total_energy first (from CSV), then fallback to energies
            if 'total_energy' in struct:
                energies.append(struct['total_energy'])
            elif struct['energies'] is not None and len(struct['energies']) > 0:
                energies.append(struct['energies'].sum() if hasattr(struct['energies'], 'sum') else struct['energies'][0])
            else:
                energies.append(0)
            sim_types.append(struct.get('sim_type', 'unknown'))
            temperatures.append(struct.get('temperature', 300))

        energies = np.array(energies)
        sim_types = np.array(sim_types)
        temperatures = np.array(temperatures)

        # Get unique sim types
        unique_sim_types = sorted(list(set(sim_types)))

        # Get colormap settings from config
        plots_config = self.config.get('plots', {})
        default_cmap = plots_config.get('colormap', 'RdBu')
        color_levels = plots_config.get('color_levels', 10)

        # Helper function to get PCA variance
        def get_var(i):
            if pca_obj is not None and i < len(pca_obj.explained_variance_ratio_):
                return 100 * pca_obj.explained_variance_ratio_[i]
            return 0

        # === Overall Analysis (3x2 grid) ===
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))

        # Row 1: PCA
        # Left: Energy-colored
        ax = axes[0, 0]
        norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                            c=energies, cmap=default_cmap, norm=norm, alpha=0.6, s=30)
        ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
        ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
        ax.set_title('PCA: Overall (colored by energy)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Energy')

        # Right: Sim type-colored
        ax = axes[0, 1]
        colors_sim = plt.cm.Set3(np.linspace(0, 1, len(unique_sim_types)))
        color_map = {t: colors_sim[i] for i, t in enumerate(unique_sim_types)}

        for sim_type in unique_sim_types:
            mask = sim_types == sim_type
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                      c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)

        ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
        ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
        ax.set_title('PCA: Overall (colored by sim type)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Row 2: t-SNE
        if tsne_result is not None:
            # Left: Energy-colored
            ax = axes[1, 0]
            norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
            scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                                c=energies, cmap=default_cmap, norm=norm, alpha=0.6, s=30)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_title('t-SNE: Overall (colored by energy)')
            ax.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Energy')

            # Right: Sim type-colored
            ax = axes[1, 1]
            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                          c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_title('t-SNE: Overall (colored by sim type)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 't-SNE not available', ha='center', va='center')
            axes[1, 0].axis('off')
            axes[1, 1].text(0.5, 0.5, 't-SNE not available', ha='center', va='center')
            axes[1, 1].axis('off')

        # Row 3: Clustering
        if labels is not None:
            # Left: Energy-colored
            ax = axes[2, 0]
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                                c=energies, cmap=default_cmap, alpha=0.6, s=30)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Clustering: Overall (colored by energy)')
            ax.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Energy')

            # Overlay cluster boundaries
            unique_labels = sorted(list(set(labels)))
            for label in unique_labels:
                mask = labels == label
                if np.sum(mask) > 0:
                    centroid = pca_result[mask].mean(axis=0)
                    ax.text(centroid[0], centroid[1], f'C{label}' if label >= 0 else 'Out',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Right: Sim type-colored
            ax = axes[2, 1]
            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                          c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Clustering: Overall (colored by sim type)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # Overlay cluster labels
            for label in unique_labels:
                mask = labels == label
                if np.sum(mask) > 0:
                    centroid = pca_result[mask].mean(axis=0)
                    ax.text(centroid[0], centroid[1], f'C{label}' if label >= 0 else 'Out',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            axes[2, 0].text(0.5, 0.5, 'Clustering not available', ha='center', va='center')
            axes[2, 0].axis('off')
            axes[2, 1].text(0.5, 0.5, 'Clustering not available', ha='center', va='center')
            axes[2, 1].axis('off')

        plt.suptitle(f'Combined Analysis - Overall ({prefix}CG Data)' if use_cg_data else 'Combined Analysis - Overall',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{prefix}combined_overall.png', bbox_inches='tight', dpi=300)
        print(f"Saved: {self.output_dir / 'figures' / f'{prefix}combined_overall.png'}")
        plt.close()

        # === Per-Sim-Type Analysis ===
        if has_temp:
            unique_temps = sorted(list(set(temperatures)))

            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                if np.sum(mask) == 0:
                    continue

                fig, axes = plt.subplots(3, 2, figsize=(14, 18))

                pca_subset = pca_result[mask]
                energies_subset = energies[mask]
                temps_subset = temperatures[mask]
                if tsne_result is not None:
                    tsne_subset = tsne_result[mask]
                else:
                    tsne_subset = None
                if labels is not None:
                    labels_subset = labels[mask]
                else:
                    labels_subset = None

                # Create 10 discrete levels for temperature
                from matplotlib.colors import BoundaryNorm
                bounds = np.linspace(min(unique_temps) - 25, max(unique_temps) + 25, color_levels + 1)
                temp_norm_disc = BoundaryNorm(bounds, ncolors=color_levels)

                # Row 1: PCA
                # Left: Energy-colored
                ax = axes[0, 0]
                norm = plt.Normalize(vmin=energies_subset.min(), vmax=energies_subset.max())
                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                    c=energies_subset, cmap=default_cmap, norm=norm, alpha=0.6, s=30)
                ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
                ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
                ax.set_title(f'PCA: {sim_type} (colored by energy)')
                ax.grid(alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Energy')

                # Right: Temperature-colored
                ax = axes[0, 1]
                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                    c=temps_subset, cmap=default_cmap, norm=temp_norm_disc, alpha=0.6, s=30)
                ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
                ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
                ax.set_title(f'PCA: {sim_type} (colored by temperature)')
                ax.grid(alpha=0.3)
                cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                if len(unique_temps) <= color_levels:
                    cbar.set_ticks(unique_temps)

                # Row 2: t-SNE
                if tsne_subset is not None:
                    # Left: Energy-colored
                    ax = axes[1, 0]
                    norm = plt.Normalize(vmin=energies_subset.min(), vmax=energies_subset.max())
                    scatter = ax.scatter(tsne_subset[:, 0], tsne_subset[:, 1],
                                        c=energies_subset, cmap=default_cmap, norm=norm, alpha=0.6, s=30)
                    ax.set_xlabel('t-SNE Component 1')
                    ax.set_ylabel('t-SNE Component 2')
                    ax.set_title(f't-SNE: {sim_type} (colored by energy)')
                    ax.grid(alpha=0.3)
                    plt.colorbar(scatter, ax=ax, label='Energy')

                    # Right: Temperature-colored
                    ax = axes[1, 1]
                    scatter = ax.scatter(tsne_subset[:, 0], tsne_subset[:, 1],
                                        c=temps_subset, cmap=default_cmap, norm=temp_norm_disc, alpha=0.6, s=30)
                    ax.set_xlabel('t-SNE Component 1')
                    ax.set_ylabel('t-SNE Component 2')
                    ax.set_title(f't-SNE: {sim_type} (colored by temperature)')
                    ax.grid(alpha=0.3)
                    cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                    if len(unique_temps) <= color_levels:
                        cbar.set_ticks(unique_temps)
                else:
                    axes[1, 0].axis('off')
                    axes[1, 1].axis('off')

                # Row 3: Clustering
                if labels_subset is not None:
                    # Left: Energy-colored
                    ax = axes[2, 0]
                    scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                        c=energies_subset, cmap=default_cmap, alpha=0.6, s=30)
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_title(f'Clustering: {sim_type} (colored by energy)')
                    ax.grid(alpha=0.3)
                    plt.colorbar(scatter, ax=ax, label='Energy')

                    # Overlay cluster labels
                    unique_labels_subset = sorted(list(set(labels_subset)))
                    for label in unique_labels_subset:
                        mask_l = labels_subset == label
                        if np.sum(mask_l) > 0:
                            centroid = pca_subset[mask_l].mean(axis=0)
                            ax.text(centroid[0], centroid[1], f'C{label}' if label >= 0 else 'Out',
                                   fontsize=10, fontweight='bold',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                    # Right: Temperature-colored
                    ax = axes[2, 1]
                    scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                        c=temps_subset, cmap=default_cmap, norm=temp_norm_disc, alpha=0.6, s=30)
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_title(f'Clustering: {sim_type} (colored by temperature)')
                    ax.grid(alpha=0.3)
                    cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                    if len(unique_temps) <= color_levels:
                        cbar.set_ticks(unique_temps)

                    # Overlay cluster labels
                    for label in unique_labels_subset:
                        mask_l = labels_subset == label
                        if np.sum(mask_l) > 0:
                            centroid = pca_subset[mask_l].mean(axis=0)
                            ax.text(centroid[0], centroid[1], f'C{label}' if label >= 0 else 'Out',
                                   fontsize=10, fontweight='bold',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                else:
                    axes[2, 0].axis('off')
                    axes[2, 1].axis('off')

                plt.suptitle(f'Combined Analysis - {sim_type} ({prefix}CG Data)' if use_cg_data else f'Combined Analysis - {sim_type}',
                            fontsize=16, fontweight='bold')
                plt.tight_layout()
                safe_sim_type = sim_type.replace('-', '_').replace('/', '_')
                plt.savefig(self.output_dir / 'figures' / f'{prefix}combined_{safe_sim_type}.png', bbox_inches='tight', dpi=300)
                print(f"Saved: {self.output_dir / 'figures' / f'{prefix}combined_{safe_sim_type}.png'}")
                plt.close()

    def extract_outliers(self, method='zscore', threshold=2.0):
        """Extract structures that are significantly different."""
        print(f"Extracting outliers using {method} (threshold={threshold})...")

        outlier_indices = []

        if method == 'zscore':
            # Use PCA space for outlier detection
            from scipy import stats
            for i in range(self.pca_result.shape[1]):
                z_scores = np.abs(stats.zscore(self.pca_result[:, i]))
                outlier_indices.extend(np.where(z_scores > threshold)[0])

        elif method == 'cluster':
            # Use cluster labels (-1 means noise/outlier)
            outlier_indices = np.where(self.labels == -1)[0]

        elif method == 'distance':
            # Use distance-based outlier detection
            from scipy.spatial.distance import pdist
            distances = pdist(self.pca_result[:, :3])
            dist_matrix = squareform(distances)
            mean_dist = dist_matrix.mean(axis=1)
            threshold_val = mean_dist.mean() + threshold * mean_dist.std()
            outlier_indices = np.where(mean_dist > threshold_val)[0]

        outlier_indices = np.unique(outlier_indices)
        print(f"Found {len(outlier_indices)} outlier structures")

        # Save outlier data to CSV
        outlier_data = []
        for idx in outlier_indices:
            struct = self.structures[idx]
            outlier_data.append({
                'index': idx,
                'timestep': struct['timestep'],
                'sim_type': struct.get('sim_type', 'unknown'),
                'source_file': struct['source_file'],
                'mean_energy': struct['energies'].mean() if struct['energies'] is not None else None,
                'std_energy': struct['energies'].std() if struct['energies'] is not None else None,
                'pc1': self.pca_result[idx, 0] if self.pca_result is not None else None,
                'pc2': self.pca_result[idx, 1] if self.pca_result is not None else None,
                'pc3': self.pca_result[idx, 2] if self.pca_result is not None else None,
                'tsne1': self.tsne_result[idx, 0] if self.tsne_result is not None else None,
                'tsne2': self.tsne_result[idx, 1] if self.tsne_result is not None else None,
                'cluster': self.labels[idx] if self.labels is not None else None
            })

        outlier_df = pd.DataFrame(outlier_data)
        outlier_df.to_csv(self.output_dir / 'outlier_structures.csv', index=False)
        print(f"Saved: {self.output_dir / 'outlier_structures.csv'}")

        # Also save full structure data for outliers
        self.save_detailed_structures(outlier_indices, 'outlier_structures_detailed.csv')

        return outlier_indices, outlier_df

    def save_detailed_structures(self, indices, filename):
        """Save detailed atomic data for specified structures."""
        detailed_data = []

        for idx in indices:
            struct = self.structures[idx]

            # Save per-atom data
            for i, atom_id in enumerate(struct['ids']):
                detailed_data.append({
                    'structure_index': idx,
                    'atom_id': atom_id,
                    'type': struct['types'][i],
                    'x': struct['positions'][i, 0],
                    'y': struct['positions'][i, 1],
                    'z': struct['positions'][i, 2],
                    'energy': struct['energies'][i] if struct['energies'] is not None else None,
                    'fx': struct['forces'][i, 0] if struct['forces'] is not None else None,
                    'fy': struct['forces'][i, 1] if struct['forces'] is not None else None,
                    'fz': struct['forces'][i, 2] if struct['forces'] is not None else None,
                    'timestep': struct['timestep'],
                    'sim_type': struct.get('sim_type', 'unknown')
                })

        df = pd.DataFrame(detailed_data)
        df.to_csv(self.output_dir / filename, index=False)
        print(f"Saved: {self.output_dir / filename}")

    def save_all_analysis_results(self):
        """Save all analysis results to files."""
        # Save PCA results
        if self.pca_result is not None:
            pca_df = pd.DataFrame(self.pca_result[:, :10],
                                  columns=[f'PC{i+1}' for i in range(10)])
            pca_df['timestep'] = [s['timestep'] for s in self.structures]
            pca_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.structures]
            if self.structures[0]['energies'] is not None:
                pca_df['mean_energy'] = [s['energies'].mean() for s in self.structures]
            pca_df.to_csv(self.output_dir / 'pca_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'pca_results.csv'}")

        # Save t-SNE results
        if self.tsne_result is not None:
            tsne_df = pd.DataFrame(self.tsne_result,
                                   columns=['tSNE1', 'tSNE2'])
            tsne_df['timestep'] = [s['timestep'] for s in self.structures]
            tsne_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.structures]
            if self.structures[0]['energies'] is not None:
                tsne_df['mean_energy'] = [s['energies'].mean() for s in self.structures]
            tsne_df.to_csv(self.output_dir / 'tsne_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'tsne_results.csv'}")

        # Save GNN results
        if self.gnn_result is not None:
            gnn_df = pd.DataFrame(self.gnn_result,
                                  columns=[f'GNN{i+1}' for i in range(self.gnn_result.shape[1])])
            gnn_df['timestep'] = [s['timestep'] for s in self.structures]
            gnn_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.structures]
            if self.structures[0]['energies'] is not None:
                gnn_df['mean_energy'] = [s['energies'].mean() for s in self.structures]
            gnn_df.to_csv(self.output_dir / 'gnn_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'gnn_results.csv'}")

        # Save descriptors
        if len(self.descriptors) > 0:
            desc_df = pd.DataFrame(self.descriptors)
            desc_df['timestep'] = [s['timestep'] for s in self.structures]
            desc_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.structures]
            desc_df.to_csv(self.output_dir / 'descriptors.csv', index=False)
            print(f"Saved: {self.output_dir / 'descriptors.csv'}")

    def run_full_analysis(self, max_frames=None, gnn_graph_viz=6, gnn_edge_cutoff=None):
        """
        Run complete structure analysis pipeline.

        Parameters
        ----------
        max_frames : int, optional
            Maximum number of frames to analyze
        gnn_graph_viz : int, optional
            Number of GNN graph structures to visualize (0 to skip)
        gnn_edge_cutoff : float, optional
            Edge cutoff distance for GNN graph visualization
        """
        print("=" * 60)
        print("ATOMIC STRUCTURE ANALYSIS")
        print("=" * 60)
        print(f"Input directory: {self.config['paths']['aa_data_base_dir']}")
        print(f"Output directory: {self.output_dir}")
        print()

        # Create figures directory
        (self.output_dir / 'figures').mkdir(exist_ok=True)

        # Load trajectories
        self.load_trajectories(max_frames=max_frames)

        if len(self.structures) == 0:
            print("No structures loaded!")
            return

        # Compute descriptors
        self.compute_descriptors()

        # Perform PCA
        self.pca_result, pca = self.perform_pca(n_components=10)
        self._pca_object = pca

        # Perform t-SNE
        self.tsne_result = self.perform_tsne(n_components=2)

        # Compute GNN embeddings
        self.compute_gnn_embeddings()

        # Cluster structures
        self.labels = self.cluster_structures(method='dbscan')

        # Generate plots
        print("\nGenerating plots...")
        self.plot_pca_analysis()
        self.plot_tsne_analysis()
        self.plot_gnn_analysis()

        # GNN graph structure visualization (new)
        if gnn_graph_viz > 0:
            print("\nGenerating GNN graph structure visualizations...")
            self.plot_gnn_graph_structure(num_examples=gnn_graph_viz, edge_cutoff=gnn_edge_cutoff)
            self.plot_gnn_node_features(num_examples=50)

        self.plot_cluster_analysis()

        # Extract outliers
        outlier_indices, outlier_df = self.extract_outliers(method='cluster', threshold=2.0)

        # Save all results
        print("\nSaving analysis results...")
        self.save_all_analysis_results()

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)

    def run_cg_analysis(self, max_frames=None, max_per_file=10):
        """
        Run CG structure analysis pipeline.

        Parameters
        ----------
        max_frames : int, optional
            Maximum total number of frames to analyze
        max_per_file : int, optional
            Maximum number of frames per trajectory file
        """
        print("=" * 60)
        print("CG STRUCTURE ANALYSIS")
        print("=" * 60)
        print(f"Input directory: {self.config.get('cg_data_base_dir', '/mnt/d/Workbench/CH_CG/02.cg_dataset')}")
        print(f"Output directory: {self.output_dir}")

        # Read max_frames and max_per_file from config if not specified
        analysis_config = self.config.get('analysis', {})
        if max_frames is None:
            max_frames = analysis_config.get('max_frames', 500)
        if max_per_file is None:
            max_per_file = analysis_config.get('max_per_file', 10)

        print(f"Max frames: {max_frames}, Max per file: {max_per_file}")
        print()

        # Create figures directory
        (self.output_dir / 'figures').mkdir(exist_ok=True)

        # Load CG trajectories
        self.load_cg_trajectories(max_frames=max_frames, max_per_file=max_per_file)

        if not hasattr(self, 'cg_structures') or len(self.cg_structures) == 0:
            print("No CG structures loaded!")
            return

        # Compute descriptors for CG structures
        self.compute_cg_descriptors()

        if self.cg_descriptors is None:
            print("Failed to compute CG descriptors!")
            return

        # Read PCA settings from config
        pca_config = self.config.get('pca', {})
        n_components = pca_config.get('n_components', 10)

        # Perform PCA on CG data
        print("Performing PCA on CG descriptors...")
        pca_cg = PCA(n_components=n_components)
        self.cg_pca_result = pca_cg.fit_transform(self.cg_descriptors)
        self._cg_pca_object = pca_cg
        print(f"PCA explained variance: {pca_cg.explained_variance_ratio_[:3]}")

        # Read t-SNE settings from config
        tsne_config = self.config.get('tsne', {})
        tsne_n_comp = tsne_config.get('n_components', 2)
        tsne_perp = tsne_config.get('perplexity', 30)

        # Perform t-SNE on CG data
        print("Performing t-SNE on CG descriptors...")
        tsne_cg = TSNE(n_components=tsne_n_comp, perplexity=tsne_perp, random_state=42, max_iter=1000)
        self.cg_tsne_result = tsne_cg.fit_transform(self.cg_descriptors)
        print(f"t-SNE result shape: {self.cg_tsne_result.shape}")

        # Read clustering settings from config
        cluster_config = self.config.get('clustering', {})
        cluster_method = cluster_config.get('method', 'dbscan')
        min_samples = cluster_config.get('min_samples', 5)
        n_clusters = cluster_config.get('n_clusters', 4)

        # Cluster CG structures
        print(f"Clustering CG structures using {cluster_method}...")
        if cluster_method == 'dbscan':
            from scipy.spatial.distance import pdist
            distances = pdist(self.cg_pca_result[:, :3])
            eps = np.percentile(distances, 30)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.cg_labels = clusterer.fit_predict(self.cg_pca_result[:, :3])
        else:  # kmeans
            n_clust = min(n_clusters, len(self.cg_pca_result) // 2)
            clusterer = KMeans(n_clusters=n_clust, random_state=42)
            self.cg_labels = clusterer.fit_predict(self.cg_pca_result[:, :3])

        n_clusters_found = len(set(self.cg_labels)) - (1 if -1 in self.cg_labels else 0)
        print(f"Found {n_clusters_found} clusters")

        # Generate plots with new format
        print("\nGenerating CG analysis plots...")
        self.plot_pca_analysis(use_cg_data=True)
        self.plot_tsne_analysis(use_cg_data=True)
        self.plot_cluster_analysis(use_cg_data=True)

        # Generate combined analysis plot
        print("\nGenerating combined analysis plot...")
        self.plot_combined_analysis(use_cg_data=True)

        # Save CG results
        print("\nSaving CG analysis results...")
        self.save_cg_results()

        print("\n" + "=" * 60)
        print("CG Analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)

    def save_cg_results(self):
        """Save CG analysis results to files."""
        if not hasattr(self, 'cg_structures') or len(self.cg_structures) == 0:
            return

        # Save PCA results
        if hasattr(self, 'cg_pca_result') and self.cg_pca_result is not None:
            pca_df = pd.DataFrame(self.cg_pca_result[:, :10],
                                  columns=[f'PC{i+1}' for i in range(10)])
            pca_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.cg_structures]
            pca_df['temperature'] = [s.get('temperature', 300) for s in self.cg_structures]
            # Use total_energy from CSV if available
            if 'total_energy' in self.cg_structures[0]:
                pca_df['total_energy'] = [s.get('total_energy', 0) for s in self.cg_structures]
            elif self.cg_structures[0]['energies'] is not None:
                pca_df['total_energy'] = [s['energies'].sum() if s['energies'] is not None else 0
                                         for s in self.cg_structures]
            pca_df.to_csv(self.output_dir / 'CG_pca_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'CG_pca_results.csv'}")

        # Save t-SNE results
        if hasattr(self, 'cg_tsne_result') and self.cg_tsne_result is not None:
            tsne_df = pd.DataFrame(self.cg_tsne_result, columns=['tSNE1', 'tSNE2'])
            tsne_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.cg_structures]
            tsne_df['temperature'] = [s.get('temperature', 300) for s in self.cg_structures]
            # Use total_energy from CSV if available
            if 'total_energy' in self.cg_structures[0]:
                tsne_df['total_energy'] = [s.get('total_energy', 0) for s in self.cg_structures]
            elif self.cg_structures[0]['energies'] is not None:
                tsne_df['total_energy'] = [s['energies'].sum() if s['energies'] is not None else 0
                                          for s in self.cg_structures]
            tsne_df.to_csv(self.output_dir / 'CG_tsne_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'CG_tsne_results.csv'}")

        # Save descriptors
        if hasattr(self, 'cg_descriptors') and self.cg_descriptors is not None:
            desc_df = pd.DataFrame(self.cg_descriptors)
            desc_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.cg_structures]
            desc_df['temperature'] = [s.get('temperature', 300) for s in self.cg_structures]
            desc_df.to_csv(self.output_dir / 'CG_descriptors.csv', index=False)
            print(f"Saved: {self.output_dir / 'CG_descriptors.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze atomic/CG structures using SOAP, PCA, t-SNE, and GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze atomic structures with default settings
  python analyze_atomic_structure.py

  # Analyze CG trajectory files
  python analyze_atomic_structure.py --cg-mode

  # Analyze CG with custom data directory
  python analyze_atomic_structure.py --cg-mode --cg-data-dir /path/to/cg/data

  # Limit number of frames for faster analysis
  python analyze_atomic_structure.py --max-frames 200 --max-per-file 5

  # Visualize more GNN graph structures with custom edge cutoff
  python analyze_atomic_structure.py --gnn-graph-viz 12 --gnn-edge-cutoff 3.5

  # Skip GNN graph structure visualization (faster)
  python analyze_atomic_structure.py --skip-gnn-viz
        """
    )

    parser.add_argument(
        '--cg-mode',
        action='store_true',
        help='Enable CG structure analysis mode (reads from 02.cg_dataset)'
    )

    parser.add_argument(
        '--cg-data-dir',
        type=str,
        default='/mnt/d/Workbench/CH_CG/02.cg_dataset',
        help='Directory containing CG trajectory data (default: 02.cg_dataset)'
    )

    parser.add_argument(
        '--aa-data-dir',
        type=str,
        default='/mnt/d/Workbench/CH_CG/01.aa',
        help='Directory containing atomic trajectory data'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for analysis results'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=500,
        help='Maximum number of frames to analyze'
    )

    parser.add_argument(
        '--max-per-file',
        type=int,
        default=10,
        help='Maximum number of frames per trajectory file (CG mode only)'
    )

    parser.add_argument(
        '--gnn-graph-viz',
        type=int,
        default=6,
        metavar='N',
        help='Number of GNN graph structures to visualize (default: 6)'
    )

    parser.add_argument(
        '--gnn-edge-cutoff',
        type=float,
        default=None,
        metavar='ANGSTROM',
        help='Edge cutoff distance for GNN graph visualization in Angstroms (default: auto)'
    )

    parser.add_argument(
        '--skip-gnn-viz',
        action='store_true',
        help='Skip GNN graph structure visualization'
    )

    args = parser.parse_args()

    # Determine mode and base directory
    mode = 'cg' if args.cg_mode else 'aa'
    base_dir = args.cg_data_dir if args.cg_mode else args.aa_data_dir

    # Create analyzer with proper mode
    analyzer = AtomicStructureAnalyzer(
        base_dir=base_dir,
        config_file=args.config,
        mode=mode
    )

    if args.output_dir:
        analyzer.output_dir = Path(args.output_dir)
        analyzer.output_dir.mkdir(parents=True, exist_ok=True)

    # Override CG data directory if specified via CLI
    if args.cg_mode and args.cg_data_dir:
        if 'paths' not in analyzer.config:
            analyzer.config['paths'] = {}
        analyzer.config['paths']['cg_data_base_dir'] = args.cg_data_dir

    # Run appropriate analysis
    if args.cg_mode:
        # CG mode: analyze CG trajectory files
        analyzer.run_cg_analysis(
            max_frames=args.max_frames,
            max_per_file=args.max_per_file
        )
    else:
        # AA mode: analyze atomic trajectory files (original behavior)
        # Pass visualization options
        analyzer.run_full_analysis(
            max_frames=args.max_frames,
            gnn_graph_viz=args.gnn_graph_viz if not args.skip_gnn_viz else 0,
            gnn_edge_cutoff=args.gnn_edge_cutoff
        )


if __name__ == '__main__':
    main()
