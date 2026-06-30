"""analyze-atomic domain logic: SOAP/PCA/t-SNE/GNN analysis of atomic & CG structures.

Migrated from legacy ``0x-analyze_atomic_structure.py``. The two duplicate LAMMPS
readers (``LAMMPSTrajectoryReader`` / ``CGTrajectoryReader``) are dropped in
favor of :class:`cglib.lammps.LammpsDumpReader`. All heavy imports
(matplotlib / scipy / sklearn / networkx / ase / dscribe / torch) are deferred
to :func:`_import_heavy_deps` so that ``import cglib.analyze_atomic`` is cheap.
Domain algorithms (descriptor / GNN / clustering / plotting) are preserved 1:1.
"""
from __future__ import annotations

import argparse
import glob
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from .lammps import LammpsDumpReader

# =============================================================================
# Heavy-dependency lazy loader
# =============================================================================

# Symbols injected by _import_heavy_deps(); None until first use.
plt: Any = None
mcolors: Any = None
BoundaryNorm: Any = None
StandardScaler: Any = None
PCA: Any = None
TSNE: Any = None
UMAP: Any = None
DBSCAN: Any = None
KMeans: Any = None
pdist: Any = None
squareform: Any = None
stats: Any = None
nx: Any = None
Atoms: Any = None
SOAP: Any = None

# Optional-dependency flags (set by _import_heavy_deps).
HAS_NETWORKX = False
HAS_ASE = False
HAS_DSCRIBE = False
HAS_TORCH = False
HAS_UMAP = False
_DEPS_IMPORTED = False


def _numeric_path_key(path: str) -> Tuple[Any, ...]:
    """Sort key that orders paths by the numeric values in their subfolders.

    Trajectory layouts such as ``1-npt/200/NPT.200.*`` yield keys like
    ``(1, 200)``, so files are grouped by simulation and then sorted by the
    temperature subfolder. Paths without any digits fall back to a lexical
    sort at the end.
    """
    nums: List[int] = []
    for part in Path(path).parts:
        digits = ''.join(c for c in part if c.isdigit())
        if digits:
            nums.append(int(digits))
    return tuple(nums) if nums else (float('inf'), path.lower())


def _import_heavy_deps() -> None:
    """Import matplotlib/scipy/sklearn/networkx/ase/dscribe on first use.

    Sets module-level globals (``plt``, ``PCA`` ...) and ``HAS_*`` flags. All
    domain code in this module references those globals, so heavy packages are
    only required when :func:`run` (or any analyzer method) is actually called.
    """
    global plt, mcolors, BoundaryNorm
    global StandardScaler, PCA, TSNE, UMAP, DBSCAN, KMeans
    global pdist, squareform, stats
    global nx, Atoms, SOAP
    global HAS_NETWORKX, HAS_ASE, HAS_DSCRIBE, HAS_TORCH, HAS_UMAP, _DEPS_IMPORTED

    if _DEPS_IMPORTED:
        return

    # matplotlib (required by every plot method) — Nature journal style
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.font_manager as _fm
    import matplotlib.pyplot as _plt
    import matplotlib.colors as _mcolors
    from matplotlib.colors import BoundaryNorm as _BoundaryNorm
    # Global rcParams: Arial sans-serif, 183 mm double-column, thin lines,
    # no top/right spine, white background, no grid.
    _plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.figsize': (10.0, 4.5),   # wider for better visibility
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'legend.frameon': False,
    })
    # Font detection: warn (once, non-fatal) if Arial/Helvetica missing so
    # users know figures fall back to DejaVu Sans.
    _available_fonts = {f.name for f in _fm.fontManager.ttflist}
    if not ({'Arial', 'Helvetica'} & _available_fonts):
        print("Note: Arial/Helvetica not installed; falling back to DejaVu Sans "
              "for publication figures.")
    plt = _plt
    mcolors = _mcolors
    BoundaryNorm = _BoundaryNorm

    # sklearn (required by descriptor + clustering pipeline)
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    from sklearn.decomposition import PCA as _PCA
    from sklearn.manifold import TSNE as _TSNE
    from sklearn.cluster import DBSCAN as _DBSCAN, KMeans as _KMeans
    StandardScaler = _StandardScaler
    PCA = _PCA
    TSNE = _TSNE
    DBSCAN = _DBSCAN
    KMeans = _KMeans

    # scipy spatial / stats
    from scipy.spatial.distance import pdist as _pdist, squareform as _squareform
    from scipy import stats as _stats
    pdist = _pdist
    squareform = _squareform
    stats = _stats

    # Optional packages
    try:
        import networkx as _nx
        nx = _nx
        HAS_NETWORKX = True
    except ImportError:
        HAS_NETWORKX = False
        print("Warning: networkx not available. Graph topology visualization will be limited.")

    try:
        from ase import Atoms as _Atoms
        Atoms = _Atoms
        HAS_ASE = True
    except ImportError:
        HAS_ASE = False
        print("Warning: ASE not available. SOAP descriptors will use fallback features.")

    try:
        from dscribe.descriptors import SOAP as _SOAP
        # Verify the SOAP API is compatible before enabling.
        try:
            _SOAP(species=[1], rcut=4.0, n_max=4, l_max=4)
            SOAP = _SOAP
            HAS_DSCRIBE = True
        except TypeError:
            SOAP = None
            HAS_DSCRIBE = False
            print("Warning: dscribe SOAP API incompatible. Will use simplified descriptors.")
    except ImportError:
        HAS_DSCRIBE = False
        print("Warning: dscribe not available. Will use simplified descriptors.")

    # PyTorch / PyTorch Geometric are not required (GNN falls back to random).
    HAS_TORCH = False

    # UMAP (optional): only required for the new UMAP step in analyze-atomic.
    # Mirrors the networkx/ase/dscribe pattern — try-import + small fit probe
    # to verify the numba JIT cache works, set HAS_UMAP accordingly.
    try:
        from umap import UMAP as _UMAP
        try:
            _UMAP(n_neighbors=5, n_components=2).fit(np.zeros((10, 4)))
            UMAP = _UMAP
            HAS_UMAP = True
        except Exception as _e:
            UMAP = None
            HAS_UMAP = False
            print(f"Warning: umap-learn installed but runtime init failed ({_e}). "
                  "UMAP step will be skipped.")
    except ImportError:
        HAS_UMAP = False
        print("Warning: umap-learn not available. UMAP step will be skipped "
              '(install with: pip install -e ".[umap]" or pip install umap-learn).')

    _DEPS_IMPORTED = True


# =============================================================================
# Nature-style palette & legend helpers
# =============================================================================

# 8-color categorical palette (ColorBrewer-style: dark-blue -> light-blue ->
# light-red -> dark-red). Used for sim_type scatter coloring so it pairs
# naturally with the RdBu_r continuous cmap on the same figures.
_NATURE_PALETTE_8 = [
    '#08306b',  # dark blue
    '#2171b5',  # medium blue
    '#6baed6',  # light blue
    '#c6dbef',  # pale blue
    '#fcbba1',  # pale red
    '#fc9272',  # light red
    '#ef3b2c',  # medium red
    '#a50f15',  # dark red
]


def _categorical_palette(n: int) -> np.ndarray:
    """Return ``n`` RGBA rows suitable for ``scatter(color=...)``.

    For ``n <= 8`` returns the first ``n`` entries of :data:`_NATURE_PALETTE_8`
    via :func:`matplotlib.colors.to_rgba`. For ``n > 8`` cycles through the
    base 8 colors so the mapping stays stable across calls.
    """
    if n <= 0:
        return np.zeros((0, 4))
    base = [_NATURE_PALETTE_8[i % len(_NATURE_PALETTE_8)] for i in range(n)]
    return np.array([mcolors.to_rgba(c) for c in base])


def _apply_categorical_legend(ax, labels: List[str],
                              colors: Optional[np.ndarray] = None,
                              title: Optional[str] = None,
                              outside: bool = True) -> None:
    """Draw a frameless categorical legend for scatter groups.

    Builds one ``Line2D`` proxy per entry in ``labels`` (in order).
    By default the legend is placed outside the axes so it never covers
    scatter points. Colors are taken from :func:`_categorical_palette` unless
    explicitly provided.
    """
    from matplotlib.lines import Line2D
    if colors is None:
        colors = _categorical_palette(len(labels))
    handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=colors[i], markeredgecolor='none',
               markersize=5, label=str(labels[i]))
        for i in range(len(labels))
    ]
    legend_kwargs = {
        'frameon': False,
        'handletextpad': 0.4,
        'labelspacing': 0.5,
        'borderaxespad': 0.0,
    }
    if title is not None:
        legend_kwargs['title'] = title
    if outside:
        ax.legend(handles=handles, loc='upper left',
                  bbox_to_anchor=(1.02, 1), **legend_kwargs)
    else:
        ax.legend(handles=handles, loc='upper right', **legend_kwargs)


def _apply_lego_legend(ax, sim_types: List[str], outside: bool = True) -> None:
    """Draw a frameless categorical legend for sim_type scatter groups."""
    _apply_categorical_legend(ax, sim_types, outside=outside)


# =============================================================================
# Structure descriptor (SOAP + rotation/translation invariants)
# =============================================================================

class StructureDescriptor:
    """Compute structure descriptors with translation/rotation invariance."""

    def __init__(self, rcut: float = 5.0, n_max: int = 8, l_max: int = 6,
                 sigma: float = 0.5):
        self.rcut = rcut
        self.n_max = n_max
        self.l_max = l_max
        self.soap: Any = None
        self._soap_params: Dict[str, Any] = {
            'species': [1, 2],
            'rcut': rcut,
            'n_max': n_max,
            'l_max': l_max,
            'sigma': sigma,
            'periodic': True,
            'average': 'inner',
        }

    def compute_relative_positions(self, positions: np.ndarray,
                                   box: np.ndarray) -> np.ndarray:
        """Center the system (translation invariance)."""
        return positions - positions.mean(axis=0)

    def compute_rotation_invariant_features(self, positions: np.ndarray,
                                            types: np.ndarray,
                                            box: np.ndarray) -> np.ndarray:
        """Pairwise distances + atom types (rotation invariant)."""
        rel_pos = self.compute_relative_positions(positions, box)
        n_atoms = len(positions)
        features: List[List[float]] = []
        box_lengths = box[:, 1] - box[:, 0]
        for i in range(n_atoms):
            for j in range(i + 1, min(i + 100, n_atoms)):
                dr = rel_pos[i] - rel_pos[j]
                dr -= np.round(dr / box_lengths) * box_lengths
                dist = float(np.linalg.norm(dr))
                features.append([dist, types[i], types[j]])
        return np.array(features)

    def _fallback_descriptor(self, frame_data: Dict[str, Any]) -> np.ndarray:
        return self.compute_rotation_invariant_features(
            frame_data['positions'], frame_data['types'], frame_data['box']
        ).flatten()[:500]

    def compute_soap_descriptor(self, frame_data: Dict[str, Any]) -> np.ndarray:
        """Compute SOAP descriptor for one frame (or fallback features)."""
        if not HAS_DSCRIBE:
            return self._fallback_descriptor(frame_data)

        if self.soap is None:
            try:
                try:
                    self.soap = SOAP(**self._soap_params)
                except TypeError:
                    self.soap = SOAP(
                        species=[1, 2], rcut=self.rcut,
                        n_max=self.n_max, l_max=self.l_max,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: SOAP initialization failed: {exc}. Using fallback features.")
                return self._fallback_descriptor(frame_data)

        box = frame_data['box']
        cell = [[box[0, 1] - box[0, 0], 0, 0],
                [0, box[1, 1] - box[1, 0], 0],
                [0, 0, box[2, 1] - box[2, 0]]]
        atoms = Atoms(
            positions=frame_data['positions'],
            numbers=frame_data['types'],
            cell=cell,
            pbc=True,
        )
        try:
            return self.soap.create(atoms).flatten()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: SOAP computation failed: {exc}. Using fallback features.")
            return self._fallback_descriptor(frame_data)


# =============================================================================
# Graph neural network embedding (torch optional)
# =============================================================================

class GraphNeuralNetwork:
    """Simple GNN for structure embedding (falls back to random if torch absent)."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, output_dim: int = 32):
        self.model = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def atoms_to_graph(self, frame_data: Dict[str, Any], feat_dim: int = 16):
        """Convert atomic structure to a PyG ``Data`` graph (or None)."""
        if not HAS_TORCH:
            return None

        import torch  # local import; module-level is too costly for non-GNN paths
        from torch_geometric.data import Data

        positions = frame_data['positions']
        types = frame_data['types']
        n_atoms = len(positions)

        node_features = []
        for i in range(n_atoms):
            feat = np.zeros(feat_dim)
            feat[types[i] - 1] = 1
            for j, pos in enumerate(positions[i]):
                feat[2 + j * 3] = np.sin(pos)
                feat[3 + j * 3] = np.cos(pos)
            node_features.append(feat)
        node_features = torch.tensor(node_features, dtype=torch.float)

        edge_index = []
        for i in range(n_atoms):
            for j in range(i + 1, min(i + 20, n_atoms)):
                edge_index.append([i, j])
                edge_index.append([j, i])
        if not edge_index:
            return None
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=node_features, edge_index=edge_index)

    def compute_embedding(self, graph) -> np.ndarray:
        """Compute graph embedding (random fallback if torch unavailable)."""
        if not HAS_TORCH or graph is None:
            return np.random.randn(32)

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, global_mean_pool
        from torch_geometric.data import Data

        input_dim = graph.x.shape[1]
        if self.model is None or not hasattr(self.model, 'conv1'):
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
            if not hasattr(graph, 'batch'):
                graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            embedding = self.model(graph)
        return embedding.numpy().flatten()


# =============================================================================
# Main analyzer
# =============================================================================

class AtomicStructureAnalyzer:
    """Main analyzer for atomic and CG structures.

    Reads its parameters from the ``analysis_atomic`` section of the unified
    config (or a legacy-style config dict). The two LAMMPS readers from the
    legacy script are replaced by :class:`cglib.lammps.LammpsDumpReader`.
    """

    def __init__(self, base_dir: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 mode: str = 'cg',
                 output_dir: Optional[str] = None):
        self.mode = mode
        self.base_dir = Path(base_dir) if base_dir else None
        self.config: Dict[str, Any] = config if config is not None else {}

        paths = self.config.get('paths', {})
        if self.base_dir is None:
            if mode == 'cg':
                self.base_dir = Path(paths.get('cg_data_base_dir',
                                               '/mnt/d/Workbench/CH_CG/02.cg_dataset'))
            else:
                self.base_dir = Path(paths.get('aa_data_base_dir',
                                               '/mnt/d/Workbench/CH_CG/01.aa'))

        # analysis_atomic is the unified-config section; fall back to flat keys
        # for backward compatibility with legacy per-tool config files.
        aa_cfg = self.config.get('analysis_atomic', {})
        analysis_cfg = self.config.get('analysis', {})

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        elif 'output_dir' in aa_cfg:
            self.output_dir = Path(aa_cfg['output_dir'])
        elif 'output_base_dir' in paths:
            self.output_dir = Path(paths['output_base_dir'])
        else:
            self.output_dir = Path(self.config.get(
                'output_base_dir', self.base_dir / 'structure_analysis_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SOAP / descriptor configuration
        soap_cfg = {**aa_cfg.get('soap', {}), **self.config.get('soap', {})}
        self.descriptor = StructureDescriptor(
            rcut=soap_cfg.get('rcut', 5.0),
            n_max=soap_cfg.get('n_max', 8),
            l_max=soap_cfg.get('l_max', 6),
            sigma=soap_cfg.get('sigma', 0.5),
        )
        self.gnn = GraphNeuralNetwork()

        # Data containers
        self.structures: List[Dict[str, Any]] = []
        self.descriptors: Optional[np.ndarray] = None
        self.embeddings: List[np.ndarray] = []
        self.pca_result: Optional[np.ndarray] = None
        self.tsne_result: Optional[np.ndarray] = None
        self.umap_result: Optional[np.ndarray] = None
        self.gnn_result: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self._pca_object: Any = None
        self._cg_pca_object: Any = None

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    def find_trajectory_files(self) -> List[str]:
        """Find LAMMPS trajectory dump files (AA mode).

        Honors ``self.base_dir`` (set from ``paths.aa_data_base_dir`` or
        ``--base-dir``) and enumerates candidates recursively so the user
        can point at any nesting level:

          - ``…/01.aa``                    (parent-level, all sims)
          - ``…/01.aa/1-npt``              (sim-level, all temperatures)
          - ``…/01.aa/1-npt/200``          (single temperature)
          - ``…/01.aa/1-npt/traj``        (a traj/ folder directly)

        This project has mixed layouts: some sims put dumps under
        ``<sim>/traj/<file>`` while others put them under
        ``<sim>/<temp>/<file>``. A name-based filter skips obvious
        non-dumps; :meth:`load_trajectories` still calls
        ``LammpsDumpReader.parse_file()`` for final content validation.
        """
        paths = self.config.get('paths', {})
        default_aa = '/mnt/d/Workbench/CH_CG/01.aa'
        root = (self.base_dir
                or Path(paths.get('aa_data_base_dir', default_aa)))
        root = Path(root)

        # Reject by name; reader does final content-based validation.
        _NON_DUMP_SUFFIXES = (
            '.restart', '.data', '.lmp', '.pb', '.pth', '.model',
            '.sbatch', '.sh', '.slurm', '.py', '.md', '.txt', '.log',
            '.json', '.yaml', '.toml', '.csv', '.raw', '.npy', '.npz',
            '.png', '.jpg', '.pdf',
        )
        _NON_DUMP_NAMES = {'log.lammps', 'in.equil', 'in.relax', 'in.nve',
                           'README'}

        def _is_candidate(p: Path) -> bool:
            if not p.is_file():
                return False
            name = p.name
            if name in _NON_DUMP_NAMES:
                return False
            if '.restart' in name:  # catches both .restart and .restart.<n>
                return False
            if name.endswith(_NON_DUMP_SUFFIXES):
                return False
            # Final content peek: every LAMMPS dump starts with this line.
            # Cheap (single read) and definitively rejects SLURM outputs,
            # job logs, etc. that slip through the name filter.
            try:
                with open(p, 'rb') as fh:
                    return fh.read(16).startswith(b'ITEM: TIMESTEP')
            except OSError:
                return False

        candidates = glob.glob(str(root / '**' / '*'), recursive=True)
        return sorted((f for f in candidates if _is_candidate(Path(f))),
                      key=_numeric_path_key)

    def find_cg_trajectory_files(self) -> List[str]:
        """Find CG ``*_cg.lammpstrj`` files (CG mode).

        Mirrors :meth:`find_trajectory_files`: honors ``self.base_dir`` and
        uses a recursive glob so ``--base-dir`` can target any nesting level:

          - ``…/02.cg_dataset``               (parent, all sims)
          - ``…/02.cg_dataset/1-npt``         (sim, all temps)
          - ``…/02.cg_dataset/1-npt/200``     (single temperature)
          - ``…/02.cg_dataset/3-upT``         (sim without temp subdirs)

        The ``*_cg.lammpstrj`` suffix is unambiguous, so no further name or
        content filtering is needed.
        """
        paths = self.config.get('paths', {})
        default_cg = '/mnt/d/Workbench/CH_CG/02.cg_dataset'
        root = (self.base_dir
                or paths.get('cg_data_base_dir')
                or self.config.get('cg_data_base_dir')
                or default_cg)
        root = Path(root)
        pattern = str(root / '**' / '*_cg.lammpstrj')
        return sorted(glob.glob(pattern, recursive=True), key=_numeric_path_key)

    @staticmethod
    def _parse_cg_filepath(filepath: str) -> Tuple[str, Optional[int]]:
        """Extract ``(sim_type, temperature)`` from a CG trajectory path.

        Handles the project's two layouts uniformly:

          - ``…/<sim>/<temp>/<file>_cg.lammpstrj``  -> ``(sim, int(temp))``
          - ``…/<sim>/<file>_cg.lammpstrj``          -> ``(sim, None)``

        ``None`` signals "no single temperature for this trajectory"
        (e.g. ``3-upT`` / ``4-dnT`` ramping runs where T varies per frame).
        """
        parts = Path(filepath).parts
        # parts[-1] = filename; look at the parent(s) for sim/temp.
        if len(parts) >= 3 and parts[-2].isdigit():
            return parts[-3], int(parts[-2])
        if len(parts) >= 2:
            return parts[-2], None
        return 'unknown', None

    @staticmethod
    def _extract_sim_type_and_temp(filepath: str) -> Tuple[str, Optional[int]]:
        """Extract ``(sim_type, nominal_temp)`` from an AA dump path.

        Robust to the project's three observed layouts:

          1. ``…/<sim>/traj/<file>``        (sim_type=<sim>, temp=None)
          2. ``…/<sim>/<temp>/<file>``      (sim_type=<sim>, temp=int)
          3. ``…/<sim>/<PREFIX>.<T>.<step>`` (sim_type=<sim>, temp from filename)

        Last resort: sim_type = parent dir basename, temp = None.
        """
        p = Path(filepath)
        parts = p.parts
        name = p.name
        parent = p.parent.name

        # Rule 1: a 'traj' segment anywhere in the path -> the parent of 'traj'
        # is the sim dir (or, for ``<sim>/<temp>/traj/<file>`` layouts, the
        # grandparent is the sim dir and the parent is the nominal T).
        if 'traj' in parts:
            try:
                idx = parts.index('traj')
                if idx >= 1:
                    parent_of_traj = parts[idx - 1]
                    if parent_of_traj.isdigit() and idx >= 2:
                        return parts[idx - 2], int(parent_of_traj)
                    return parent_of_traj, None
            except ValueError:
                pass

        # Rule 2: parent dir is a pure integer -> that's the temp
        if parent.isdigit():
            sim_type = parts[-3] if len(parts) >= 3 else 'unknown'
            return sim_type, int(parent)

        # Rule 3: filename has shape '<PREFIX>.<temp>.<step>' with integer
        # temp and step fields.
        tokens = name.split('.')
        if len(tokens) >= 3 and tokens[-2].isdigit():
            try:
                return parent, int(tokens[-2])
            except ValueError:
                pass

        # Fallback
        return parent, None

    @staticmethod
    def _build_structure_id(sim_type: str, temp: Optional[int],
                            timestep: int) -> str:
        """Build a human-readable composite ID ``{sim}/{temp_or_ramp}@{step}``.

        ``temp=None`` (ramping runs) renders as the literal ``ramp``.
        """
        temp_str = str(temp) if temp is not None else 'ramp'
        return f"{sim_type}/{temp_str}@{timestep}"

    def _propagate_metadata(self) -> None:
        """Ensure every loaded structure has the canonical metadata fields.

        Guarantees ``sim_type``, ``temp`` (Optional[int]), ``timestep``,
        ``source_file``, and ``structure_id`` exist on each entry of
        ``self.structures`` (AA) and ``self.cg_structures`` (CG). Called once
        at the end of each loader; idempotent.
        """
        for pool_name in ('structures', 'cg_structures'):
            pool = getattr(self, pool_name, None)
            if not pool:
                continue
            for s in pool:
                # If either sim_type or temp is missing, re-derive both from
                # source_file (they are coupled; partial extraction is wrong).
                if ('sim_type' not in s or 'temp' not in s) and s.get('source_file'):
                    sim, temp = self._extract_sim_type_and_temp(s['source_file'])
                    s.setdefault('sim_type', sim)
                    s.setdefault('temp', temp)
                # Final fallbacks for missing keys
                s.setdefault('sim_type', 'unknown')
                s.setdefault('temp', None)
                s.setdefault('timestep', 0)
                if not s.get('structure_id'):
                    s['structure_id'] = self._build_structure_id(
                        s['sim_type'], s['temp'], s['timestep'])

    # ------------------------------------------------------------------
    # Trajectory loading
    # ------------------------------------------------------------------
    def load_trajectories(self, max_frames: Optional[int] = None
                          ) -> List[Dict[str, Any]]:
        """Load one frame per LAMMPS dump file (AA mode)."""
        files = self.find_trajectory_files()
        print(f"Found {len(files)} trajectory files")

        aa_cfg = self.config.get('analysis_atomic', {})
        analysis_cfg = self.config.get('analysis', {})
        if max_frames is None:
            max_frames = (self.config.get('data_loading', {}).get('max_frames')
                          or aa_cfg.get('max_frames')
                          or analysis_cfg.get('max_frames', 500))

        structures: List[Dict[str, Any]] = []
        for filepath in files:
            if len(structures) >= max_frames:
                break
            try:
                reader = LammpsDumpReader(filepath)
                if not reader.parse_file():
                    continue
                frame = reader.read_first_frame()
                if frame is None:
                    continue
                frame['source_file'] = filepath
                sim_type, temp = self._extract_sim_type_and_temp(filepath)
                frame['sim_type'] = sim_type
                frame['temp'] = temp
                # Numeric temperature alias for legacy plotting code that reads
                # 'temperature'; ramping runs fall back to 300 K (TODO: should
                # pull measured T from log.lammps in a follow-up).
                frame['temperature'] = temp if temp is not None else 300
                structures.append(frame)
                if len(structures) % 50 == 0:
                    print(f"  Loaded {len(structures)} frames...")
            except Exception as exc:  # noqa: BLE001
                print(f"  Warning: Failed to load {filepath}: {exc}")

        print(f"Successfully loaded {len(structures)} frames")
        self.structures = structures
        self._propagate_metadata()
        return structures

    def load_cg_trajectories(self, max_frames: Optional[int] = None,
                             max_per_file: Optional[int] = None
                             ) -> List[Dict[str, Any]]:
        """Load CG trajectory frames (CG mode)."""
        files = self.find_cg_trajectory_files()

        # Filter by enabled simulation names.
        allowed_names = {sim['name'] for sim in self.config.get('simulations', [])
                         if sim.get('enabled', True)}
        if allowed_names:
            files = [f for f in files
                     if self._parse_cg_filepath(f)[0] in allowed_names]
            print(f"Found {len(files)} CG trajectory files "
                  f"(filtered by allowed types: {sorted(allowed_names)})")
        else:
            print(f"Found {len(files)} CG trajectory files")

        aa_cfg = self.config.get('analysis_atomic', {})
        analysis_cfg = self.config.get('analysis', {})
        if max_frames is None:
            max_frames = (self.config.get('data_loading', {}).get('max_frames')
                          or aa_cfg.get('max_frames')
                          or analysis_cfg.get('max_frames', 500))
        if max_per_file is None:
            max_per_file = (self.config.get('data_loading', {}).get('max_per_file')
                            or aa_cfg.get('max_per_file')
                            or analysis_cfg.get('max_per_file', 10))

        cg_data: List[Dict[str, Any]] = []
        for filepath in files:
            if len(cg_data) >= max_frames:
                break

            sim_type, parsed_temp = self._parse_cg_filepath(filepath)
            # Canonical: Optional[int], None for ramping runs.
            temp = parsed_temp
            # Legacy numeric alias (TODO: real fix pulls measured T from log).
            temperature_numeric = parsed_temp if parsed_temp is not None else 300

            # Optional energy lookup from sibling _particles.csv
            basename = Path(filepath).stem.replace('_cg', '')
            particles_csv = Path(filepath).parent / f"{basename}_particles.csv"
            total_energy: Optional[float] = None
            if particles_csv.exists():
                try:
                    df = pd.read_csv(particles_csv)
                    if 'c_pe' in df.columns:
                        total_energy = float(df['c_pe'].sum())
                except Exception:  # noqa: BLE001
                    pass

            try:
                reader = LammpsDumpReader(filepath)
                if not reader.parse_file():
                    continue
                frames = reader.read_all_frames()

                count_for_this_file = 0
                for frame in frames:
                    if len(cg_data) >= max_frames:
                        break
                    if count_for_this_file >= max_per_file:
                        break
                    frame['source_file'] = filepath
                    frame['sim_type'] = sim_type
                    frame['temp'] = temp
                    frame['temperature'] = temperature_numeric
                    if total_energy is not None:
                        frame['total_energy'] = total_energy
                        frame['energies'] = np.array([total_energy])
                    cg_data.append(frame)
                    count_for_this_file += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  Warning: Failed to load {filepath}: {exc}")

        print(f"Successfully loaded {len(cg_data)} CG frames from {len(files)} files")
        self.cg_structures = cg_data
        self._propagate_metadata()
        return cg_data

    # ------------------------------------------------------------------
    # Descriptor / embedding computation
    # ------------------------------------------------------------------
    def compute_descriptors(self) -> Optional[np.ndarray]:
        print("Computing structure descriptors...")
        descriptors = []
        for i, struct in enumerate(self.structures):
            descriptors.append(self.descriptor.compute_soap_descriptor(struct))
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(self.structures)} structures...")
        self.descriptors = np.array(descriptors)
        print(f"Descriptor shape: {self.descriptors.shape}")
        self.descriptors = StandardScaler().fit_transform(self.descriptors)
        return self.descriptors

    def compute_cg_descriptors(self, cg_data: Optional[List[Dict[str, Any]]] = None
                               ) -> Optional[np.ndarray]:
        if cg_data is None:
            cg_data = getattr(self, 'cg_structures', None)
        if not cg_data:
            print("No CG structures available")
            return None

        print("Computing CG structure descriptors...")
        descriptors = []
        for i, struct in enumerate(cg_data):
            descriptors.append(self.descriptor.compute_soap_descriptor(struct))
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(cg_data)} structures...")
        self.cg_descriptors = np.array(descriptors)
        print(f"CG descriptor shape: {self.cg_descriptors.shape}")
        self.cg_descriptors = StandardScaler().fit_transform(self.cg_descriptors)
        return self.cg_descriptors

    def compute_gnn_embeddings(self) -> np.ndarray:
        if not HAS_TORCH:
            print("GNN not available, using random embeddings")
            self.gnn_result = np.random.randn(len(self.structures), 32)
            return self.gnn_result

        print("Computing GNN embeddings...")
        embeddings = []
        for i, struct in enumerate(self.structures):
            graph = self.gnn.atoms_to_graph(struct)
            embeddings.append(self.gnn.compute_embedding(graph))
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(self.structures)} structures...")
        self.gnn_result = np.array(embeddings)
        print(f"GNN embedding shape: {self.gnn_result.shape}")
        return self.gnn_result

    # ------------------------------------------------------------------
    # PCA / t-SNE / clustering
    # ------------------------------------------------------------------
    def perform_pca(self, n_components: int = 3) -> Tuple[np.ndarray, Any]:
        print(f"Performing PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(self.descriptors)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
        return self.pca_result, pca

    def perform_tsne(self, n_components: int = 2, perplexity: float = 30,
                     max_iter: int = 1000) -> np.ndarray:
        print(f"Performing t-SNE (n_components={n_components}, perplexity={perplexity})...")
        tsne = TSNE(n_components=n_components, perplexity=perplexity,
                    random_state=42, max_iter=max_iter, verbose=1)
        self.tsne_result = tsne.fit_transform(self.descriptors)
        print(f"t-SNE result shape: {self.tsne_result.shape}")
        return self.tsne_result

    def perform_umap(self, n_components: int = 5, n_neighbors: int = 15,
                     min_dist: float = 0.1, metric: str = 'euclidean',
                     random_state: int = 42) -> Optional[np.ndarray]:
        """UMAP 降维。umap-learn 未安装时返回 None 并打印 warning。"""
        if not HAS_UMAP or UMAP is None:
            print("UMAP not available (install umap-learn); skipping UMAP step.")
            return None
        print(f"Performing UMAP (n_components={n_components}, "
              f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")
        reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                       min_dist=min_dist, metric=metric,
                       random_state=random_state)
        self.umap_result = reducer.fit_transform(self.descriptors)
        print(f"UMAP result shape: {self.umap_result.shape}")
        return self.umap_result

    def cluster_structures(self, method: str = 'kmeans') -> np.ndarray:
        aa_cfg = self.config.get('analysis_atomic', {})
        cluster_cfg = {**aa_cfg.get('clustering', {}), **self.config.get('clustering', {})}
        analysis_cfg = self.config.get('analysis', {})
        min_samples = (analysis_cfg.get('min_samples')
                       or cluster_cfg.get('min_samples', 5))
        n_clusters = cluster_cfg.get('n_clusters', 4)
        space = cluster_cfg.get('space', 'pca')

        # Select the projection to cluster in. Defaults to PCA[:, :3]; if the
        # requested space is unavailable (e.g. UMAP skipped due to missing dep),
        # warn and fall back to PCA.
        if space == 'umap':
            if getattr(self, 'umap_result', None) is None:
                print(f"Warning: clustering.space='umap' but no UMAP result; "
                      f"falling back to PCA.")
                proj = self.pca_result[:, :3]
            else:
                proj = self.umap_result
        elif space == 'tsne':
            if self.tsne_result is None:
                print(f"Warning: clustering.space='tsne' but no t-SNE result; "
                      f"falling back to PCA.")
                proj = self.pca_result[:, :3]
            else:
                proj = self.tsne_result
        else:
            proj = self.pca_result[:, :3]
        print(f"Clustering on {space} projection (shape={proj.shape})")

        if method == 'dbscan':
            print("Clustering using DBSCAN...")
            distances = pdist(proj)
            eps = np.percentile(distances, 30)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            print("Clustering using K-Means...")
            n_clust = min(n_clusters, len(proj) // 2)
            clusterer = KMeans(n_clusters=n_clust, random_state=42)

        self.labels = clusterer.fit_predict(proj)
        n_found = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        print(f"Found {n_found} clusters")
        return self.labels

    # ------------------------------------------------------------------
    # Internal helpers used by plot methods
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_metadata(structures: List[Dict[str, Any]]
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        energies: List[float] = []
        sim_types: List[str] = []
        temperatures: List[int] = []
        for struct in structures:
            if 'total_energy' in struct:
                energies.append(struct['total_energy'])
            elif struct['energies'] is not None and len(struct['energies']) > 0:
                energies.append(struct['energies'].sum()
                                if hasattr(struct['energies'], 'sum')
                                else struct['energies'][0])
            else:
                energies.append(0)
            sim_types.append(struct.get('sim_type', 'unknown'))
            temperatures.append(struct.get('temperature', 300))
        return (np.array(energies), np.array(sim_types), np.array(temperatures))

    @staticmethod
    def _extract_id_columns(structures: List[Dict[str, Any]]
                            ) -> Dict[str, List[Any]]:
        """Build the canonical tracing columns for CSV writers.

        Returns a dict with keys ``structure_id``, ``source_file``,
        ``sim_type``, ``temp``, ``timestep`` — each a list aligned with
        ``structures``. ``temp`` is ``None`` for ramping runs; callers that
        want a numeric value should fall back to ``structure.get('temperature', 300)``.
        """
        cols: Dict[str, List[Any]] = {
            'structure_id': [], 'source_file': [], 'sim_type': [],
            'temp': [], 'timestep': [],
        }
        for s in structures:
            cols['structure_id'].append(s.get('structure_id', ''))
            cols['source_file'].append(s.get('source_file', ''))
            cols['sim_type'].append(s.get('sim_type', 'unknown'))
            cols['temp'].append(s.get('temp'))
            cols['timestep'].append(s.get('timestep', 0))
        return cols

    def _pca_variances(self) -> np.ndarray:
        if self._pca_object is not None:
            return self._pca_object.explained_variance_ratio_
        return np.ones(10) / 10

    def _cg_pca_variances(self) -> np.ndarray:
        if self._cg_pca_object is not None:
            return self._cg_pca_object.explained_variance_ratio_
        return np.ones(10) / 10

    # ------------------------------------------------------------------
    # Plot methods (1:1 from legacy)
    # ------------------------------------------------------------------
    def plot_pca_analysis(self, use_cg_data: bool = False) -> None:
        if use_cg_data and getattr(self, 'cg_pca_result', None) is not None:
            pca_result = self.cg_pca_result
            structures = getattr(self, 'cg_structures', [])
            prefix = 'CG_'
            pca_obj = self._cg_pca_object
        else:
            pca_result = self.pca_result
            structures = self.structures
            prefix = ''
            pca_obj = self._pca_object

        if pca_result is None:
            print("No PCA results available")
            return

        has_temp = len(structures) > 0 and 'temperature' in structures[0]
        energies, sim_types, temperatures = self._extract_metadata(structures)
        unique_sim_types = sorted(list(set(sim_types)))

        def get_var(i: int) -> float:
            if pca_obj is not None and i < len(pca_obj.explained_variance_ratio_):
                return 100 * pca_obj.explained_variance_ratio_[i]
            return 0

        # === Overall ===
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))

        ax = axes[0]
        norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                             c=energies, cmap='RdBu_r', norm=norm, alpha=0.6, s=30)
        ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
        ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
        ax.set_title('Overall: PC1 vs PC2 (colored by energy)')
        plt.colorbar(scatter, ax=ax, label='Energy')

        ax = axes[1]
        colors_sim = _categorical_palette(len(unique_sim_types))
        color_map = {t: colors_sim[i] for i, t in enumerate(unique_sim_types)}
        for sim_type in unique_sim_types:
            mask = sim_types == sim_type
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
        ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
        ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
        ax.set_title('Overall: PC1 vs PC2 (colored by sim type)')
        _apply_lego_legend(ax, unique_sim_types, outside=True)

        plt.suptitle(f'PCA Analysis - Overall ({prefix}CG Data)' if use_cg_data
                     else 'PCA Analysis - Overall',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = self.output_dir / 'figures' / f'{prefix}pca_overall.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

        # === Per sim type ===
        if has_temp:
            unique_temps = sorted(list(set(temperatures)))
            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                if np.sum(mask) == 0:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
                pca_subset = pca_result[mask]
                energies_subset = energies[mask]
                temps_subset = temperatures[mask]

                ax = axes[0]
                norm = plt.Normalize(vmin=energies_subset.min(), vmax=energies_subset.max())
                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                     c=energies_subset, cmap='RdBu_r', norm=norm,
                                     alpha=0.6, s=30)
                ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
                ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
                ax.set_title(f'{sim_type}: PC1 vs PC2 (colored by energy)')
                plt.colorbar(scatter, ax=ax, label='Energy')

                ax = axes[1]
                colors_temp = _categorical_palette(len(unique_temps))
                temp_color_map = {t: colors_temp[i] for i, t in enumerate(unique_temps)}
                for t in unique_temps:
                    mask = temps_subset == t
                    if np.sum(mask) == 0:
                        continue
                    ax.scatter(pca_subset[mask, 0], pca_subset[mask, 1],
                               c=[temp_color_map[t]], label=str(int(t)), alpha=0.6, s=30)
                ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
                ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
                ax.set_title(f'{sim_type}: PC1 vs PC2 (colored by temperature)')
                _apply_categorical_legend(ax, [str(int(t)) for t in unique_temps],
                                          colors=colors_temp, title='Temperature (K)',
                                          outside=True)

                plt.suptitle(f'PCA Analysis - {sim_type} ({prefix}CG Data)' if use_cg_data
                             else f'PCA Analysis - {sim_type}',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                safe = sim_type.replace('-', '_').replace('/', '_')
                out_path = self.output_dir / 'figures' / f'{prefix}pca_{safe}.png'
                plt.savefig(out_path, bbox_inches='tight', dpi=300)
                print(f"Saved: {out_path}")
                plt.close()

    def plot_tsne_analysis(self, use_cg_data: bool = False) -> None:
        if use_cg_data and getattr(self, 'cg_tsne_result', None) is not None:
            tsne_result = self.cg_tsne_result
            structures = getattr(self, 'cg_structures', [])
            prefix = 'CG_'
        else:
            tsne_result = self.tsne_result
            structures = self.structures
            prefix = ''

        if tsne_result is None:
            print("No t-SNE results available")
            return

        has_temp = len(structures) > 0 and 'temperature' in structures[0]
        energies, sim_types, temperatures = self._extract_metadata(structures)
        unique_sim_types = sorted(list(set(sim_types)))

        # === Overall ===
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))

        ax = axes[0]
        norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                             c=energies, cmap='RdBu_r', norm=norm, alpha=0.6, s=30)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('Overall: t-SNE (colored by energy)')
        plt.colorbar(scatter, ax=ax, label='Energy')

        ax = axes[1]
        colors_sim = _categorical_palette(len(unique_sim_types))
        color_map = {t: colors_sim[i] for i, t in enumerate(unique_sim_types)}
        for sim_type in unique_sim_types:
            mask = sim_types == sim_type
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                       c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('Overall: t-SNE (colored by sim type)')
        _apply_lego_legend(ax, unique_sim_types, outside=True)

        plt.suptitle(f't-SNE Analysis - Overall ({prefix}CG Data)' if use_cg_data
                     else 't-SNE Analysis - Overall',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = self.output_dir / 'figures' / f'{prefix}tsne_overall.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

        # === Per sim type ===
        if has_temp:
            unique_temps = sorted(list(set(temperatures)))
            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                if np.sum(mask) == 0:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
                tsne_subset = tsne_result[mask]
                energies_subset = energies[mask]
                temps_subset = temperatures[mask]

                ax = axes[0]
                norm = plt.Normalize(vmin=energies_subset.min(), vmax=energies_subset.max())
                scatter = ax.scatter(tsne_subset[:, 0], tsne_subset[:, 1],
                                     c=energies_subset, cmap='RdBu_r', norm=norm,
                                     alpha=0.6, s=30)
                ax.set_xlabel('t-SNE Component 1')
                ax.set_ylabel('t-SNE Component 2')
                ax.set_title(f'{sim_type}: t-SNE (colored by energy)')
                plt.colorbar(scatter, ax=ax, label='Energy')

                ax = axes[1]
                colors_temp = _categorical_palette(len(unique_temps))
                temp_color_map = {t: colors_temp[i] for i, t in enumerate(unique_temps)}
                for t in unique_temps:
                    mask = temps_subset == t
                    if np.sum(mask) == 0:
                        continue
                    ax.scatter(tsne_subset[mask, 0], tsne_subset[mask, 1],
                               c=[temp_color_map[t]], label=str(int(t)), alpha=0.6, s=30)
                ax.set_xlabel('t-SNE Component 1')
                ax.set_ylabel('t-SNE Component 2')
                ax.set_title(f'{sim_type}: t-SNE (colored by temperature)')
                _apply_categorical_legend(ax, [str(int(t)) for t in unique_temps],
                                          colors=colors_temp, title='Temperature (K)',
                                          outside=True)

                plt.suptitle(f't-SNE Analysis - {sim_type} ({prefix}CG Data)' if use_cg_data
                             else f't-SNE Analysis - {sim_type}',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                safe = sim_type.replace('-', '_').replace('/', '_')
                out_path = self.output_dir / 'figures' / f'{prefix}tsne_{safe}.png'
                plt.savefig(out_path, bbox_inches='tight', dpi=300)
                print(f"Saved: {out_path}")
                plt.close()

    def plot_umap_analysis(self, use_cg_data: bool = False) -> None:
        if use_cg_data and getattr(self, 'cg_umap_result', None) is not None:
            umap_result = self.cg_umap_result
            structures = getattr(self, 'cg_structures', [])
            prefix = 'CG_'
        else:
            umap_result = getattr(self, 'umap_result', None)
            structures = self.structures
            prefix = ''

        if umap_result is None:
            print("No UMAP results available")
            return

        has_temp = len(structures) > 0 and 'temperature' in structures[0]
        energies, sim_types, temperatures = self._extract_metadata(structures)
        unique_sim_types = sorted(list(set(sim_types)))

        # === Overall ===
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))

        ax = axes[0]
        norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
        scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1],
                             c=energies, cmap='RdBu_r', norm=norm, alpha=0.6, s=30)
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_title('Overall: UMAP (colored by energy)')
        plt.colorbar(scatter, ax=ax, label='Energy')

        ax = axes[1]
        colors_sim = _categorical_palette(len(unique_sim_types))
        color_map = {t: colors_sim[i] for i, t in enumerate(unique_sim_types)}
        for sim_type in unique_sim_types:
            mask = sim_types == sim_type
            ax.scatter(umap_result[mask, 0], umap_result[mask, 1],
                       c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_title('Overall: UMAP (colored by sim type)')
        _apply_lego_legend(ax, unique_sim_types, outside=True)

        plt.suptitle(f'UMAP Analysis - Overall ({prefix}CG Data)' if use_cg_data
                     else 'UMAP Analysis - Overall',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = self.output_dir / 'figures' / f'{prefix}umap_overall.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

        # === Per sim type ===
        if has_temp:
            unique_temps = sorted(list(set(temperatures)))
            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                if np.sum(mask) == 0:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
                umap_subset = umap_result[mask]
                energies_subset = energies[mask]
                temps_subset = temperatures[mask]

                ax = axes[0]
                norm = plt.Normalize(vmin=energies_subset.min(), vmax=energies_subset.max())
                scatter = ax.scatter(umap_subset[:, 0], umap_subset[:, 1],
                                     c=energies_subset, cmap='RdBu_r', norm=norm,
                                     alpha=0.6, s=30)
                ax.set_xlabel('UMAP Component 1')
                ax.set_ylabel('UMAP Component 2')
                ax.set_title(f'{sim_type}: UMAP (colored by energy)')
                plt.colorbar(scatter, ax=ax, label='Energy')

                ax = axes[1]
                colors_temp = _categorical_palette(len(unique_temps))
                temp_color_map = {t: colors_temp[i] for i, t in enumerate(unique_temps)}
                for t in unique_temps:
                    mask = temps_subset == t
                    if np.sum(mask) == 0:
                        continue
                    ax.scatter(umap_subset[mask, 0], umap_subset[mask, 1],
                               c=[temp_color_map[t]], label=str(int(t)), alpha=0.6, s=30)
                ax.set_xlabel('UMAP Component 1')
                ax.set_ylabel('UMAP Component 2')
                ax.set_title(f'{sim_type}: UMAP (colored by temperature)')
                _apply_categorical_legend(ax, [str(int(t)) for t in unique_temps],
                                          colors=colors_temp, title='Temperature (K)',
                                          outside=True)

                plt.suptitle(f'UMAP Analysis - {sim_type} ({prefix}CG Data)' if use_cg_data
                             else f'UMAP Analysis - {sim_type}',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                safe = sim_type.replace('-', '_').replace('/', '_')
                out_path = self.output_dir / 'figures' / f'{prefix}umap_{safe}.png'
                plt.savefig(out_path, bbox_inches='tight', dpi=300)
                print(f"Saved: {out_path}")
                plt.close()

    def plot_gnn_analysis(self) -> None:
        if self.gnn_result is None or self.gnn_result.shape[1] < 2:
            print("No GNN results available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))

        energies: List[float] = []
        sim_types: List[str] = []
        for struct in self.structures:
            if struct['energies'] is not None:
                energies.append(struct['energies'].mean())
            else:
                energies.append(0)
            sim_types.append(struct.get('sim_type', 'unknown'))
        energies_arr = np.array(energies)
        norm = plt.Normalize(vmin=energies_arr.min(), vmax=energies_arr.max())

        ax = axes[0]
        scatter = ax.scatter(self.gnn_result[:, 0], self.gnn_result[:, 1],
                             c=energies_arr, cmap='RdBu_r', norm=norm, alpha=0.6, s=30)
        ax.set_xlabel('GNN Dimension 1')
        ax.set_ylabel('GNN Dimension 2')
        ax.set_title('GNN Embedding: Colored by Energy')
        plt.colorbar(scatter, ax=ax, label='Energy')

        ax = axes[1]
        unique_types = list(set(sim_types))
        colors_type = _categorical_palette(len(unique_types))
        color_map = {t: colors_type[i] for i, t in enumerate(unique_types)}
        for sim_type in unique_types:
            mask = np.array(sim_types) == sim_type
            ax.scatter(self.gnn_result[mask, 0], self.gnn_result[mask, 1],
                       c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
        ax.set_xlabel('GNN Dimension 1')
        ax.set_ylabel('GNN Dimension 2')
        ax.set_title('GNN Embedding: Colored by Simulation Type')
        _apply_lego_legend(ax, unique_types, outside=len(unique_types) > 6)

        plt.suptitle('GNN Analysis of Atomic Structures',
                     fontsize=14, fontweight='bold')
        out_path = self.output_dir / 'figures' / 'gnn_analysis.png'
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()

    def plot_gnn_graph_structure(self, num_examples: int = 6,
                                 edge_cutoff: Optional[float] = None) -> None:
        if not HAS_TORCH:
            print("GNN graph visualization requires PyTorch/PyG")
            return

        print(f"Visualizing GNN graph structure for {num_examples} examples...")
        if self.pca_result is not None:
            indices = np.linspace(0, len(self.structures) - 1, num_examples, dtype=int)
        else:
            indices = range(min(num_examples, len(self.structures)))

        n_cols = 3
        n_rows = (num_examples + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(10.0, 2.4 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

        for plot_idx, struct_idx in enumerate(indices):
            if plot_idx >= num_examples:
                break
            struct = self.structures[struct_idx]
            row = plot_idx // n_cols
            col = plot_idx % n_cols

            graph = self.gnn.atoms_to_graph(struct)
            if graph is None:
                continue

            positions = struct['positions']
            types = struct['types']
            n_atoms = len(positions)

            if edge_cutoff is None:
                box = struct['box']
                box_lengths = box[:, 1] - box[:, 0]
                edge_cutoff = min(box_lengths) * 0.3

            edges: List[Tuple[int, int]] = []
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dr = positions[i] - positions[j]
                    box_lengths = struct['box'][:, 1] - struct['box'][:, 0]
                    dr -= np.round(dr / box_lengths) * box_lengths
                    dist = np.linalg.norm(dr)
                    if dist < edge_cutoff:
                        edges.append((i, j))

            ax = fig.add_subplot(gs[row, col], projection='3d')
            colors = ['#1f77b4' if t == 1 else '#ff7f0e' for t in types]
            sizes = [100 if t == 1 else 50 for t in types]
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidth=0.5)
            for (i, j) in edges:
                pos_i, pos_j = positions[i], positions[j]
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], [pos_i[2], pos_j[2]],
                        'gray', alpha=0.3, linewidth=0.5)

            energy = struct['energies'].mean() if struct['energies'] is not None else None
            sim_type = struct.get('sim_type', 'unknown')
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            if isinstance(energy, float):
                ax.set_title(f'#{struct_idx} | {sim_type}\n'
                             f'Atoms: {n_atoms} | Edges: {len(edges)}\n'
                             f'E: {energy:.2f} eV', fontsize=10)
            else:
                ax.set_title(f'#{struct_idx} | {sim_type}\n'
                             f'Atoms: {n_atoms} | Edges: {len(edges)}', fontsize=10)

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
        out_path = self.output_dir / 'figures' / 'gnn_graph_structure.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

        if HAS_NETWORKX:
            indices_list = list(indices)[:min(4, len(indices))]
            self.plot_gnn_network_topology(indices_list)

    def plot_gnn_network_topology(self, indices) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0))
        axes_flat = axes.flatten()

        for ax_idx, struct_idx in enumerate(indices):
            if ax_idx >= 4:
                break
            ax = axes_flat[ax_idx]
            struct = self.structures[struct_idx]

            G = nx.Graph()
            positions = struct['positions']
            types = struct['types']
            n_atoms = len(positions)
            for i in range(n_atoms):
                G.add_node(i, atom_type=types[i],
                           pos=(positions[i, 0], positions[i, 1], positions[i, 2]))
            box_lengths = struct['box'][:, 1] - struct['box'][:, 0]
            edge_cutoff = min(box_lengths) * 0.3
            for i in range(n_atoms):
                for j in range(i + 1, min(i + 20, n_atoms)):
                    dr = positions[i] - positions[j]
                    dr -= np.round(dr / box_lengths) * box_lengths
                    dist = np.linalg.norm(dr)
                    if dist < edge_cutoff:
                        G.add_edge(i, j, weight=dist)

            pos_2d = nx.spring_layout(G, seed=42, k=1 / np.sqrt(n_atoms))
            node_colors = ['#1f77b4' if G.nodes[i]['atom_type'] == 1 else '#ff7f0e'
                           for i in G.nodes()]
            node_sizes = [200 if G.nodes[i]['atom_type'] == 1 else 100 for i in G.nodes()]
            nx.draw_networkx_nodes(G, pos_2d, ax=ax, node_color=node_colors,
                                   node_size=node_sizes, alpha=0.8, edgecolors='black')
            nx.draw_networkx_edges(G, pos_2d, ax=ax, alpha=0.3, width=0.5)
            if n_atoms <= 20:
                nx.draw_networkx_labels(G, pos_2d, ax=ax, font_size=8)

            degrees = dict(G.degree())
            avg_degree = np.mean(list(degrees.values()))
            clustering = nx.average_clustering(G)
            ax.set_title(f'#{struct_idx} | {struct.get("sim_type", "unknown")}\n'
                         f'Nodes: {n_atoms}, Edges: {G.number_of_edges()}\n'
                         f'Avg Degree: {avg_degree:.2f}, Clustering: {clustering:.3f}')
            ax.axis('off')

        plt.suptitle('GNN Graph Topology (2D Spring Layout)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = self.output_dir / 'figures' / 'gnn_network_topology.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

    def plot_gnn_node_features(self, num_examples: int = 50) -> None:
        if not HAS_TORCH or len(self.structures) == 0:
            print("Cannot visualize node features")
            return

        print("Analyzing GNN node features...")
        all_features: List[np.ndarray] = []
        all_types: List[int] = []
        n_sample = min(num_examples, len(self.structures))
        indices = np.linspace(0, len(self.structures) - 1, n_sample, dtype=int)

        for idx in indices:
            struct = self.structures[idx]
            graph = self.gnn.atoms_to_graph(struct)
            if graph is not None:
                all_features.append(graph.x.numpy())
                all_types.extend(struct['types'])

        if not all_features:
            print("No valid graphs found")
            return
        features_arr = np.vstack(all_features)
        types_arr = np.array(all_types)

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0))

        ax = axes[0, 0]
        for atom_type in [1, 2]:
            mask = types_arr == atom_type
            if mask.any():
                feature_means = features_arr[mask].mean(axis=0)
                ax.plot(feature_means, label=f'Type {atom_type} ({"C" if atom_type == 1 else "H"})',
                        marker='o')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean Feature Value')
        ax.set_title('Node Feature Profiles by Atom Type')
        ax.legend()

        ax = axes[0, 1]
        n_show = min(50, features_arr.shape[0])
        im = ax.imshow(features_arr[:n_show].T, aspect='auto', cmap='RdBu_r')
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Feature Index')
        ax.set_title('Node Feature Heatmap')
        plt.colorbar(im, ax=ax, label='Feature Value')

        ax = axes[1, 0]
        feature_var = features_arr.var(axis=0)
        ax.bar(range(len(feature_var)), feature_var, alpha=0.7)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Variance')
        ax.set_title('Feature Variance Across All Nodes')

        ax = axes[1, 1]
        n_feat_sample = min(16, features_arr.shape[1])
        corr = np.corrcoef(features_arr[:, :n_feat_sample].T)
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        ax.set_title(f'Feature Correlation Matrix (first {n_feat_sample} features)')
        plt.colorbar(im, ax=ax, label='Correlation')

        plt.suptitle('GNN Node Feature Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = self.output_dir / 'figures' / 'gnn_node_features.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

    def plot_cluster_analysis(self, use_cg_data: bool = False) -> None:
        if use_cg_data and getattr(self, 'cg_labels', None) is not None:
            labels = self.cg_labels
            pca_result = getattr(self, 'cg_pca_result', None)
            structures = getattr(self, 'cg_structures', [])
            prefix = 'CG_'
        else:
            labels = self.labels
            pca_result = self.pca_result
            structures = self.structures
            prefix = ''

        if labels is None or pca_result is None:
            print("No clustering / PCA results available")
            return

        has_temp = len(structures) > 0 and 'temperature' in structures[0]
        energies, sim_types, temperatures = self._extract_metadata(structures)
        unique_sim_types = sorted(list(set(sim_types)))
        unique_labels = sorted(list(set(labels)))

        def _draw_centroids(ax, label_arr, pca_arr):
            for label in sorted(set(label_arr)):
                mask = label_arr == label
                if np.sum(mask) > 0:
                    centroid = pca_arr[mask].mean(axis=0)
                    ax.text(centroid[0], centroid[1],
                            f'C{label}' if label >= 0 else 'Out',
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # === Overall ===
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))

        ax = axes[0]
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                             c=energies, cmap='RdBu_r', alpha=0.6, s=30)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Overall: Clusters (colored by energy)')
        plt.colorbar(scatter, ax=ax, label='Energy')
        _draw_centroids(ax, labels, pca_result)

        ax = axes[1]
        colors_sim = _categorical_palette(len(unique_sim_types))
        color_map = {t: colors_sim[i] for i, t in enumerate(unique_sim_types)}
        for sim_type in unique_sim_types:
            mask = sim_types == sim_type
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Overall: Clusters (colored by sim type)')
        _apply_lego_legend(ax, unique_sim_types, outside=len(unique_sim_types) > 6)
        _draw_centroids(ax, labels, pca_result)

        plt.suptitle(f'Cluster Analysis - Overall ({prefix}CG Data)' if use_cg_data
                     else 'Cluster Analysis - Overall',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = self.output_dir / 'figures' / f'{prefix}cluster_overall.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

        # === Per sim type ===
        if has_temp:
            unique_temps = sorted(list(set(temperatures)))
            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                if np.sum(mask) == 0:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
                pca_subset = pca_result[mask]
                energies_subset = energies[mask]
                temps_subset = temperatures[mask]
                labels_subset = labels[mask]

                ax = axes[0]
                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                     c=energies_subset, cmap='RdBu_r', alpha=0.6, s=30)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title(f'{sim_type}: Clusters (colored by energy)')
                plt.colorbar(scatter, ax=ax, label='Energy')
                _draw_centroids(ax, labels_subset, pca_subset)

                ax = axes[1]
                n_temp_levels = len(set(temps_subset))
                bounds = np.linspace(min(unique_temps) - 25, max(unique_temps) + 25,
                                     n_temp_levels + 1)
                temp_norm_disc = BoundaryNorm(bounds, ncolors=n_temp_levels)
                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                     c=temps_subset, cmap=plt.cm.RdBu_r,
                                     norm=temp_norm_disc, alpha=0.6, s=30)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title(f'{sim_type}: Clusters (colored by temperature)')
                cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                if len(unique_temps) <= 10:
                    cbar.set_ticks(unique_temps)
                _draw_centroids(ax, labels_subset, pca_subset)

                plt.suptitle(f'Cluster Analysis - {sim_type} ({prefix}CG Data)' if use_cg_data
                             else f'Cluster Analysis - {sim_type}',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                safe = sim_type.replace('-', '_').replace('/', '_')
                out_path = self.output_dir / 'figures' / f'{prefix}cluster_{safe}.png'
                plt.savefig(out_path, bbox_inches='tight', dpi=300)
                print(f"Saved: {out_path}")
                plt.close()

        # Unused-but-kept for parity (suppresses linter noise around unique_labels).
        del unique_labels

    def plot_combined_analysis(self, use_cg_data: bool = False) -> None:
        """3x2 grid: PCA / t-SNE / Clustering rows, energy / sim-type columns."""
        if use_cg_data and getattr(self, 'cg_pca_result', None) is not None:
            pca_result = self.cg_pca_result
            tsne_result = getattr(self, 'cg_tsne_result', None)
            labels = getattr(self, 'cg_labels', None)
            structures = getattr(self, 'cg_structures', [])
            prefix = 'CG_'
            pca_obj = self._cg_pca_object
        else:
            pca_result = self.pca_result
            tsne_result = self.tsne_result
            labels = self.labels
            structures = self.structures
            prefix = ''
            pca_obj = self._pca_object

        if pca_result is None:
            print("No PCA results available for combined analysis")
            return

        has_temp = len(structures) > 0 and 'temperature' in structures[0]
        energies, sim_types, temperatures = self._extract_metadata(structures)
        unique_sim_types = sorted(list(set(sim_types)))

        aa_cfg = self.config.get('analysis_atomic', {})
        plots_cfg = self.config.get('plots', {})
        default_cmap = plots_cfg.get('colormap', 'RdBu_r')
        color_levels = plots_cfg.get('color_levels',
                                     aa_cfg.get('color_levels', 10))

        def get_var(i: int) -> float:
            if pca_obj is not None and i < len(pca_obj.explained_variance_ratio_):
                return 100 * pca_obj.explained_variance_ratio_[i]
            return 0

        # === Overall (3x2) ===
        fig, axes = plt.subplots(3, 2, figsize=(10.0, 9.0))

        # Row 1: PCA
        ax = axes[0, 0]
        norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                             c=energies, cmap=default_cmap, norm=norm, alpha=0.6, s=30)
        ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
        ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
        ax.set_title('PCA: Overall (colored by energy)')
        plt.colorbar(scatter, ax=ax, label='Energy')

        ax = axes[0, 1]
        colors_sim = _categorical_palette(len(unique_sim_types))
        color_map = {t: colors_sim[i] for i, t in enumerate(unique_sim_types)}
        for sim_type in unique_sim_types:
            mask = sim_types == sim_type
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
        ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
        ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
        ax.set_title('PCA: Overall (colored by sim type)')
        ax.legend(fontsize=8)

        # Row 2: t-SNE
        if tsne_result is not None:
            ax = axes[1, 0]
            norm = plt.Normalize(vmin=energies.min(), vmax=energies.max())
            scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                                 c=energies, cmap=default_cmap, norm=norm, alpha=0.6, s=30)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_title('t-SNE: Overall (colored by energy)')
            plt.colorbar(scatter, ax=ax, label='Energy')

            ax = axes[1, 1]
            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                           c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_title('t-SNE: Overall (colored by sim type)')
            ax.legend(fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 't-SNE not available', ha='center', va='center')
            axes[1, 0].axis('off')
            axes[1, 1].text(0.5, 0.5, 't-SNE not available', ha='center', va='center')
            axes[1, 1].axis('off')

        # Row 3: Clustering
        if labels is not None:
            ax = axes[2, 0]
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                                 c=energies, cmap=default_cmap, alpha=0.6, s=30)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Clustering: Overall (colored by energy)')
            plt.colorbar(scatter, ax=ax, label='Energy')
            for label in sorted(set(labels)):
                mask = labels == label
                if np.sum(mask) > 0:
                    centroid = pca_result[mask].mean(axis=0)
                    ax.text(centroid[0], centroid[1],
                            f'C{label}' if label >= 0 else 'Out',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax = axes[2, 1]
            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                           c=[color_map[sim_type]], label=sim_type, alpha=0.6, s=30)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Clustering: Overall (colored by sim type)')
            ax.legend(fontsize=8)
            for label in sorted(set(labels)):
                mask = labels == label
                if np.sum(mask) > 0:
                    centroid = pca_result[mask].mean(axis=0)
                    ax.text(centroid[0], centroid[1],
                            f'C{label}' if label >= 0 else 'Out',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            axes[2, 0].text(0.5, 0.5, 'Clustering not available', ha='center', va='center')
            axes[2, 0].axis('off')
            axes[2, 1].text(0.5, 0.5, 'Clustering not available', ha='center', va='center')
            axes[2, 1].axis('off')

        plt.suptitle(f'Combined Analysis - Overall ({prefix}CG Data)' if use_cg_data
                     else 'Combined Analysis - Overall',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        out_path = self.output_dir / 'figures' / f'{prefix}combined_overall.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
        plt.close()

        # === Per sim type ===
        if has_temp:
            unique_temps = sorted(list(set(temperatures)))
            for sim_type in unique_sim_types:
                mask = sim_types == sim_type
                if np.sum(mask) == 0:
                    continue

                fig, axes = plt.subplots(3, 2, figsize=(10.0, 9.0))
                pca_subset = pca_result[mask]
                energies_subset = energies[mask]
                temps_subset = temperatures[mask]
                tsne_subset = tsne_result[mask] if tsne_result is not None else None
                labels_subset = labels[mask] if labels is not None else None

                # Match discrete color levels to the number of temperature
                # subfolders / distinct temperatures for this simulation.
                color_levels = len(set(temps_subset))
                bounds = np.linspace(min(unique_temps) - 25, max(unique_temps) + 25,
                                     color_levels + 1)
                temp_norm_disc = BoundaryNorm(bounds, ncolors=color_levels)

                # Row 1: PCA
                ax = axes[0, 0]
                norm = plt.Normalize(vmin=energies_subset.min(), vmax=energies_subset.max())
                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                     c=energies_subset, cmap=default_cmap, norm=norm,
                                     alpha=0.6, s=30)
                ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
                ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
                ax.set_title(f'PCA: {sim_type} (colored by energy)')
                plt.colorbar(scatter, ax=ax, label='Energy')

                ax = axes[0, 1]
                scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                     c=temps_subset, cmap=default_cmap,
                                     norm=temp_norm_disc, alpha=0.6, s=30)
                ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
                ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
                ax.set_title(f'PCA: {sim_type} (colored by temperature)')
                cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                if len(unique_temps) <= color_levels:
                    cbar.set_ticks(unique_temps)

                # Row 2: t-SNE
                if tsne_subset is not None:
                    ax = axes[1, 0]
                    norm = plt.Normalize(vmin=energies_subset.min(), vmax=energies_subset.max())
                    scatter = ax.scatter(tsne_subset[:, 0], tsne_subset[:, 1],
                                         c=energies_subset, cmap=default_cmap, norm=norm,
                                         alpha=0.6, s=30)
                    ax.set_xlabel('t-SNE Component 1')
                    ax.set_ylabel('t-SNE Component 2')
                    ax.set_title(f't-SNE: {sim_type} (colored by energy)')
                    plt.colorbar(scatter, ax=ax, label='Energy')

                    ax = axes[1, 1]
                    scatter = ax.scatter(tsne_subset[:, 0], tsne_subset[:, 1],
                                         c=temps_subset, cmap=default_cmap,
                                         norm=temp_norm_disc, alpha=0.6, s=30)
                    ax.set_xlabel('t-SNE Component 1')
                    ax.set_ylabel('t-SNE Component 2')
                    ax.set_title(f't-SNE: {sim_type} (colored by temperature)')
                    cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                    if len(unique_temps) <= color_levels:
                        cbar.set_ticks(unique_temps)
                else:
                    axes[1, 0].axis('off')
                    axes[1, 1].axis('off')

                # Row 3: Clustering
                if labels_subset is not None:
                    ax = axes[2, 0]
                    scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                         c=energies_subset, cmap=default_cmap,
                                         alpha=0.6, s=30)
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_title(f'Clustering: {sim_type} (colored by energy)')
                    plt.colorbar(scatter, ax=ax, label='Energy')
                    for label in sorted(set(labels_subset)):
                        mask_l = labels_subset == label
                        if np.sum(mask_l) > 0:
                            centroid = pca_subset[mask_l].mean(axis=0)
                            ax.text(centroid[0], centroid[1],
                                    f'C{label}' if label >= 0 else 'Out',
                                    fontsize=10, fontweight='bold',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                    ax = axes[2, 1]
                    scatter = ax.scatter(pca_subset[:, 0], pca_subset[:, 1],
                                         c=temps_subset, cmap=default_cmap,
                                         norm=temp_norm_disc, alpha=0.6, s=30)
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_title(f'Clustering: {sim_type} (colored by temperature)')
                    cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                    if len(unique_temps) <= color_levels:
                        cbar.set_ticks(unique_temps)
                    for label in sorted(set(labels_subset)):
                        mask_l = labels_subset == label
                        if np.sum(mask_l) > 0:
                            centroid = pca_subset[mask_l].mean(axis=0)
                            ax.text(centroid[0], centroid[1],
                                    f'C{label}' if label >= 0 else 'Out',
                                    fontsize=10, fontweight='bold',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                else:
                    axes[2, 0].axis('off')
                    axes[2, 1].axis('off')

                plt.suptitle(f'Combined Analysis - {sim_type} ({prefix}CG Data)' if use_cg_data
                             else f'Combined Analysis - {sim_type}',
                             fontsize=16, fontweight='bold')
                plt.tight_layout()
                safe = sim_type.replace('-', '_').replace('/', '_')
                out_path = self.output_dir / 'figures' / f'{prefix}combined_{safe}.png'
                plt.savefig(out_path, bbox_inches='tight', dpi=300)
                print(f"Saved: {out_path}")
                plt.close()

    # ------------------------------------------------------------------
    # Outlier extraction & result export
    # ------------------------------------------------------------------
    def extract_outliers(self, method: str = 'zscore',
                         threshold: float = 2.0
                         ) -> Tuple[np.ndarray, pd.DataFrame]:
        print(f"Extracting outliers using {method} (threshold={threshold})...")
        outlier_indices: List[int] = []

        if method == 'zscore':
            for i in range(self.pca_result.shape[1]):
                z_scores = np.abs(stats.zscore(self.pca_result[:, i]))
                outlier_indices.extend(np.where(z_scores > threshold)[0])
        elif method == 'cluster':
            outlier_indices = list(np.where(self.labels == -1)[0])
        elif method == 'distance':
            distances = pdist(self.pca_result[:, :3])
            dist_matrix = squareform(distances)
            mean_dist = dist_matrix.mean(axis=1)
            threshold_val = mean_dist.mean() + threshold * mean_dist.std()
            outlier_indices = list(np.where(mean_dist > threshold_val)[0])

        outlier_indices = list(np.unique(outlier_indices))
        print(f"Found {len(outlier_indices)} outlier structures")

        rows = []
        for idx in outlier_indices:
            struct = self.structures[idx]
            rows.append({
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
                'cluster': self.labels[idx] if self.labels is not None else None,
                'structure_id': struct.get('structure_id', ''),
                'temp': struct.get('temp'),
            })
        outlier_df = pd.DataFrame(rows)
        outlier_df.to_csv(self.output_dir / 'outlier_structures.csv', index=False)
        print(f"Saved: {self.output_dir / 'outlier_structures.csv'}")
        self.save_detailed_structures(outlier_indices, 'outlier_structures_detailed.csv')
        return np.array(outlier_indices), outlier_df

    def save_detailed_structures(self, indices, filename: str) -> None:
        rows = []
        for idx in indices:
            struct = self.structures[idx]
            for i, atom_id in enumerate(struct['ids']):
                rows.append({
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
                    'sim_type': struct.get('sim_type', 'unknown'),
                    'structure_id': struct.get('structure_id', ''),
                    'source_file': struct.get('source_file', ''),
                    'temp': struct.get('temp'),
                })
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / filename, index=False)
        print(f"Saved: {self.output_dir / filename}")

    def save_all_analysis_results(self) -> None:
        id_cols = self._extract_id_columns(self.structures)

        if self.pca_result is not None:
            pca_df = pd.DataFrame(self.pca_result[:, :10],
                                  columns=[f'PC{i+1}' for i in range(10)])
            pca_df['timestep'] = [s['timestep'] for s in self.structures]
            pca_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.structures]
            if self.structures[0]['energies'] is not None:
                pca_df['mean_energy'] = [s['energies'].mean() for s in self.structures]
            pca_df['structure_id'] = id_cols['structure_id']
            pca_df['source_file'] = id_cols['source_file']
            pca_df['temp'] = id_cols['temp']
            pca_df.to_csv(self.output_dir / 'pca_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'pca_results.csv'}")

        if self.tsne_result is not None:
            tsne_df = pd.DataFrame(self.tsne_result, columns=['tSNE1', 'tSNE2'])
            tsne_df['timestep'] = [s['timestep'] for s in self.structures]
            tsne_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.structures]
            if self.structures[0]['energies'] is not None:
                tsne_df['mean_energy'] = [s['energies'].mean() for s in self.structures]
            tsne_df['structure_id'] = id_cols['structure_id']
            tsne_df['source_file'] = id_cols['source_file']
            tsne_df['temp'] = id_cols['temp']
            tsne_df.to_csv(self.output_dir / 'tsne_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'tsne_results.csv'}")

        if getattr(self, 'umap_result', None) is not None:
            n_umap = self.umap_result.shape[1]
            umap_df = pd.DataFrame(self.umap_result,
                                   columns=[f'UMAP{i+1}' for i in range(n_umap)])
            umap_df['timestep'] = [s['timestep'] for s in self.structures]
            umap_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.structures]
            if self.structures[0]['energies'] is not None:
                umap_df['mean_energy'] = [s['energies'].mean() for s in self.structures]
            umap_df['structure_id'] = id_cols['structure_id']
            umap_df['source_file'] = id_cols['source_file']
            umap_df['temp'] = id_cols['temp']
            umap_df.to_csv(self.output_dir / 'umap_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'umap_results.csv'}")

        if self.gnn_result is not None:
            gnn_df = pd.DataFrame(self.gnn_result,
                                  columns=[f'GNN{i+1}' for i in range(self.gnn_result.shape[1])])
            gnn_df['timestep'] = [s['timestep'] for s in self.structures]
            gnn_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.structures]
            if self.structures[0]['energies'] is not None:
                gnn_df['mean_energy'] = [s['energies'].mean() for s in self.structures]
            gnn_df['structure_id'] = id_cols['structure_id']
            gnn_df['source_file'] = id_cols['source_file']
            gnn_df['temp'] = id_cols['temp']
            gnn_df.to_csv(self.output_dir / 'gnn_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'gnn_results.csv'}")

        if len(self.descriptors) > 0:
            desc_df = pd.DataFrame(self.descriptors)
            desc_df['timestep'] = [s['timestep'] for s in self.structures]
            desc_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.structures]
            desc_df['structure_id'] = id_cols['structure_id']
            desc_df['source_file'] = id_cols['source_file']
            desc_df['temp'] = id_cols['temp']
            desc_df.to_csv(self.output_dir / 'descriptors.csv', index=False)
            print(f"Saved: {self.output_dir / 'descriptors.csv'}")

    def save_cg_results(self) -> None:
        if not getattr(self, 'cg_structures', None):
            return

        id_cols = self._extract_id_columns(self.cg_structures)

        if getattr(self, 'cg_pca_result', None) is not None:
            pca_df = pd.DataFrame(self.cg_pca_result[:, :10],
                                  columns=[f'PC{i+1}' for i in range(10)])
            pca_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.cg_structures]
            pca_df['temperature'] = [s.get('temperature', 300) for s in self.cg_structures]
            if 'total_energy' in self.cg_structures[0]:
                pca_df['total_energy'] = [s.get('total_energy', 0) for s in self.cg_structures]
            elif self.cg_structures[0]['energies'] is not None:
                pca_df['total_energy'] = [s['energies'].sum() if s['energies'] is not None else 0
                                          for s in self.cg_structures]
            pca_df['structure_id'] = id_cols['structure_id']
            pca_df['source_file'] = id_cols['source_file']
            pca_df['temp'] = id_cols['temp']
            pca_df.to_csv(self.output_dir / 'CG_pca_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'CG_pca_results.csv'}")

        if getattr(self, 'cg_tsne_result', None) is not None:
            tsne_df = pd.DataFrame(self.cg_tsne_result, columns=['tSNE1', 'tSNE2'])
            tsne_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.cg_structures]
            tsne_df['temperature'] = [s.get('temperature', 300) for s in self.cg_structures]
            if 'total_energy' in self.cg_structures[0]:
                tsne_df['total_energy'] = [s.get('total_energy', 0) for s in self.cg_structures]
            elif self.cg_structures[0]['energies'] is not None:
                tsne_df['total_energy'] = [s['energies'].sum() if s['energies'] is not None else 0
                                           for s in self.cg_structures]
            tsne_df['structure_id'] = id_cols['structure_id']
            tsne_df['source_file'] = id_cols['source_file']
            tsne_df['temp'] = id_cols['temp']
            tsne_df.to_csv(self.output_dir / 'CG_tsne_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'CG_tsne_results.csv'}")

        if getattr(self, 'cg_umap_result', None) is not None:
            n_umap = self.cg_umap_result.shape[1]
            umap_df = pd.DataFrame(self.cg_umap_result,
                                   columns=[f'UMAP{i+1}' for i in range(n_umap)])
            umap_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.cg_structures]
            umap_df['temperature'] = [s.get('temperature', 300) for s in self.cg_structures]
            if 'total_energy' in self.cg_structures[0]:
                umap_df['total_energy'] = [s.get('total_energy', 0) for s in self.cg_structures]
            elif self.cg_structures[0]['energies'] is not None:
                umap_df['total_energy'] = [s['energies'].sum() if s['energies'] is not None else 0
                                           for s in self.cg_structures]
            umap_df['structure_id'] = id_cols['structure_id']
            umap_df['source_file'] = id_cols['source_file']
            umap_df['temp'] = id_cols['temp']
            umap_df.to_csv(self.output_dir / 'CG_umap_results.csv', index=False)
            print(f"Saved: {self.output_dir / 'CG_umap_results.csv'}")

        if getattr(self, 'cg_descriptors', None) is not None:
            desc_df = pd.DataFrame(self.cg_descriptors)
            desc_df['sim_type'] = [s.get('sim_type', 'unknown') for s in self.cg_structures]
            desc_df['temperature'] = [s.get('temperature', 300) for s in self.cg_structures]
            desc_df['structure_id'] = id_cols['structure_id']
            desc_df['source_file'] = id_cols['source_file']
            desc_df['temp'] = id_cols['temp']
            desc_df.to_csv(self.output_dir / 'CG_descriptors.csv', index=False)
            print(f"Saved: {self.output_dir / 'CG_descriptors.csv'}")

    # ------------------------------------------------------------------
    # Top-level pipelines
    # ------------------------------------------------------------------
    def run_full_analysis(self, max_frames: Optional[int] = None,
                          gnn_graph_viz: int = 6,
                          gnn_edge_cutoff: Optional[float] = None) -> None:
        print("=" * 60)
        print("ATOMIC STRUCTURE ANALYSIS")
        print("=" * 60)
        print(f"Input directory: {self.config.get('paths', {}).get('aa_data_base_dir', self.base_dir)}")
        print(f"Output directory: {self.output_dir}")
        print()

        (self.output_dir / 'figures').mkdir(exist_ok=True)

        self.load_trajectories(max_frames=max_frames)
        if len(self.structures) == 0:
            print("No structures loaded!")
            return

        self.compute_descriptors()
        self.pca_result, pca = self.perform_pca(n_components=10)
        self._pca_object = pca
        self.tsne_result = self.perform_tsne(n_components=2)

        aa_cfg_umap = self.config.get('analysis_atomic', {})
        umap_cfg = {**aa_cfg_umap.get('umap', {}), **self.config.get('umap', {})}
        self.umap_result = self.perform_umap(
            n_components=umap_cfg.get('n_components', 5),
            n_neighbors=umap_cfg.get('n_neighbors', 15),
            min_dist=umap_cfg.get('min_dist', 0.1),
            metric=umap_cfg.get('metric', 'euclidean'),
        )

        self.compute_gnn_embeddings()
        self.labels = self.cluster_structures(method='dbscan')

        print("\nGenerating plots...")
        self.plot_pca_analysis()
        self.plot_tsne_analysis()
        self.plot_umap_analysis()
        self.plot_gnn_analysis()

        if gnn_graph_viz > 0:
            print("\nGenerating GNN graph structure visualizations...")
            self.plot_gnn_graph_structure(num_examples=gnn_graph_viz,
                                          edge_cutoff=gnn_edge_cutoff)
            self.plot_gnn_node_features(num_examples=50)

        self.plot_cluster_analysis()
        self.extract_outliers(method='cluster', threshold=2.0)

        print("\nSaving analysis results...")
        self.save_all_analysis_results()

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)

    def run_cg_analysis(self, max_frames: Optional[int] = None,
                        max_per_file: Optional[int] = 10) -> None:
        print("=" * 60)
        print("CG STRUCTURE ANALYSIS")
        print("=" * 60)
        print(f"Input directory: {self.config.get('paths', {}).get('cg_data_base_dir', self.base_dir)}")
        print(f"Output directory: {self.output_dir}")

        aa_cfg = self.config.get('analysis_atomic', {})
        analysis_cfg = self.config.get('analysis', {})
        if max_frames is None:
            max_frames = aa_cfg.get('max_frames', analysis_cfg.get('max_frames', 500))
        if max_per_file is None:
            max_per_file = aa_cfg.get('max_per_file', analysis_cfg.get('max_per_file', 10))
        print(f"Max frames: {max_frames}, Max per file: {max_per_file}\n")

        (self.output_dir / 'figures').mkdir(exist_ok=True)

        self.load_cg_trajectories(max_frames=max_frames, max_per_file=max_per_file)
        if not getattr(self, 'cg_structures', None):
            print("No CG structures loaded!")
            return

        self.compute_cg_descriptors()
        if self.cg_descriptors is None:
            print("Failed to compute CG descriptors!")
            return

        aa_cfg = self.config.get('analysis_atomic', {})
        pca_cfg = {**aa_cfg.get('pca', {}), **self.config.get('pca', {})}
        tsne_cfg = {**aa_cfg.get('tsne', {}), **self.config.get('tsne', {})}
        umap_cfg = {**aa_cfg.get('umap', {}), **self.config.get('umap', {})}
        cluster_cfg = {**aa_cfg.get('clustering', {}), **self.config.get('clustering', {})}

        n_components = pca_cfg.get('n_components', 10)
        print("Performing PCA on CG descriptors...")
        pca_cg = PCA(n_components=n_components)
        self.cg_pca_result = pca_cg.fit_transform(self.cg_descriptors)
        self._cg_pca_object = pca_cg
        print(f"PCA explained variance: {pca_cg.explained_variance_ratio_[:3]}")

        tsne_n_comp = tsne_cfg.get('n_components', 2)
        tsne_perp = tsne_cfg.get('perplexity', 30)
        tsne_iter = tsne_cfg.get('max_iter', 1000)
        print("Performing t-SNE on CG descriptors...")
        tsne_cg = TSNE(n_components=tsne_n_comp, perplexity=tsne_perp,
                       random_state=42, max_iter=tsne_iter)
        self.cg_tsne_result = tsne_cg.fit_transform(self.cg_descriptors)
        print(f"t-SNE result shape: {self.cg_tsne_result.shape}")

        umap_n_comp = umap_cfg.get('n_components', 5)
        umap_nn = umap_cfg.get('n_neighbors', 15)
        umap_md = umap_cfg.get('min_dist', 0.1)
        umap_metric = umap_cfg.get('metric', 'euclidean')
        if HAS_UMAP and UMAP is not None:
            print("Performing UMAP on CG descriptors...")
            umap_cg = UMAP(n_components=umap_n_comp, n_neighbors=umap_nn,
                           min_dist=umap_md, metric=umap_metric, random_state=42)
            self.cg_umap_result = umap_cg.fit_transform(self.cg_descriptors)
            print(f"UMAP result shape: {self.cg_umap_result.shape}")
        else:
            print("Skipping UMAP (umap-learn not available)")

        cluster_method = cluster_cfg.get('method', 'dbscan')
        min_samples = cluster_cfg.get('min_samples', 5)
        n_clusters = cluster_cfg.get('n_clusters', 4)
        space = cluster_cfg.get('space', 'pca')

        # Select projection space for clustering (mirrors cluster_structures AA logic)
        if space == 'umap':
            if getattr(self, 'cg_umap_result', None) is not None:
                proj = self.cg_umap_result
            else:
                print(f"Warning: clustering.space='umap' but no CG UMAP result; "
                      f"falling back to PCA.")
                proj = self.cg_pca_result[:, :3]
        elif space == 'tsne':
            if getattr(self, 'cg_tsne_result', None) is not None:
                proj = self.cg_tsne_result
            else:
                print(f"Warning: clustering.space='tsne' but no CG t-SNE result; "
                      f"falling back to PCA.")
                proj = self.cg_pca_result[:, :3]
        else:  # 'pca' or unknown
            proj = self.cg_pca_result[:, :3]
        print(f"Clustering CG structures using {cluster_method} on {space} "
              f"projection (shape={proj.shape})...")
        if cluster_method == 'dbscan':
            distances = pdist(proj)
            eps = np.percentile(distances, 30)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.cg_labels = clusterer.fit_predict(proj)
        else:
            n_clust = min(n_clusters, len(proj) // 2)
            clusterer = KMeans(n_clusters=n_clust, random_state=42)
            self.cg_labels = clusterer.fit_predict(proj)
        n_found = len(set(self.cg_labels)) - (1 if -1 in self.cg_labels else 0)
        print(f"Found {n_found} clusters")

        print("\nGenerating CG analysis plots...")
        self.plot_pca_analysis(use_cg_data=True)
        self.plot_tsne_analysis(use_cg_data=True)
        self.plot_umap_analysis(use_cg_data=True)
        self.plot_cluster_analysis(use_cg_data=True)

        print("\nGenerating combined analysis plot...")
        self.plot_combined_analysis(use_cg_data=True)

        print("\nSaving CG analysis results...")
        self.save_cg_results()

        print("\n" + "=" * 60)
        print("CG Analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================

def run(config: Dict[str, Any], args: Optional[argparse.Namespace] = None) -> int:
    """cgkit analyze-atomic entry point.

    Reads parameters from ``config['analysis_atomic']`` (with CLI overrides via
    ``args``). Mode ``cg`` analyses CG trajectories (``*_cg.lammpstrj``);
    mode ``aa`` analyses atomic LAMMPS dumps.
    """
    _import_heavy_deps()

    aa_cfg = config.get('analysis_atomic', {})
    paths = config.get('paths', {})

    mode = (getattr(args, 'atomic_mode', None)
            or getattr(args, 'mode', None)
            or aa_cfg.get('mode', 'cg'))
    mode = mode if mode in ('cg', 'aa') else 'cg'

    if mode == 'aa':
        base_dir = paths.get('aa_data_base_dir', '/mnt/d/Workbench/CH_CG/01.aa')
    else:
        base_dir = paths.get('cg_data_base_dir', '/mnt/d/Workbench/CH_CG/02.cg_dataset')

    output_dir = aa_cfg.get('output_dir', paths.get('analysis_output_base_dir'))
    if output_dir is not None:
        output_dir = str(output_dir)

    # CLI override for clustering projection space (--cluster-space).
    # We mutate a shallow copy of config so the original is untouched; both
    # run_full_analysis (AA) and run_cg_analysis (CG) read this field.
    cluster_space_cli = getattr(args, 'cluster_space', None)
    if cluster_space_cli:
        config = dict(config)
        aa_cfg_copy = dict(config.get('analysis_atomic', {}))
        cluster_copy = dict(aa_cfg_copy.get('clustering', {}))
        cluster_copy['space'] = cluster_space_cli
        aa_cfg_copy['clustering'] = cluster_copy
        config['analysis_atomic'] = aa_cfg_copy

    analyzer = AtomicStructureAnalyzer(
        base_dir=base_dir,
        config=config,
        mode=mode,
        output_dir=output_dir,
    )

    max_frames = (getattr(args, 'max_frames', None) or aa_cfg.get('max_frames', 500))
    if mode == 'cg':
        max_per_file = (getattr(args, 'max_per_file', None)
                        or aa_cfg.get('max_per_file', 10))
        analyzer.run_cg_analysis(max_frames=max_frames, max_per_file=max_per_file)
    else:
        skip_viz = aa_cfg.get('skip_gnn_viz', False)
        gnn_graph_viz = 0 if skip_viz else aa_cfg.get('gnn_graph_viz', 6)
        gnn_edge_cutoff = aa_cfg.get('gnn_edge_cutoff')
        analyzer.run_full_analysis(max_frames=max_frames,
                                   gnn_graph_viz=gnn_graph_viz,
                                   gnn_edge_cutoff=gnn_edge_cutoff)
    return 0
