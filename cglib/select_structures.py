"""select-structures domain logic: cluster PCA/t-SNE/UMAP results and pick N per cluster.

Consumes a ``pca_results.csv`` / ``tsne_results.csv`` / ``umap_results.csv``
(or the CG variants) produced by :mod:`cglib.analyze_atomic`, re-clusters in
the chosen projection space (KMeans or DBSCAN, parameters user-controlled),
then for each cluster selects ``N`` maximally-spread structures via
**maximin (farthest-point) sampling**. Each selected frame is extracted from
its source LAMMPS dump into a standalone dump file in the output directory,
plus a ``selection_manifest.csv`` records every pick for traceability.

Design notes
------------
- Heavy deps (``scikit-learn``, ``scipy``) are imported lazily inside the
  functions that need them, so ``import cglib.select_structures`` stays cheap
  and does not require the analysis extras to be installed until ``run()``
  is actually called.
- Clustering is re-run on the projection coordinates (not inherited from
  ``analyze-atomic``) so the user can sweep ``--n-clusters`` / ``--eps``
  without re-running the expensive SOAP→PCA pipeline.
- Frame extraction groups selected rows by ``source_file`` and parses each
  source dump **at most once**, even when multiple picks come from the same file.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .lammps import LammpsDumpReader, write_lammps_frame
from .paths import ensure_dir


# Matplotlib is deferred — populated by _import_deps(). Keeps
# ``import cglib.select_structures`` cheap and headless-safe.
plt = None
mcolors = None


def _import_deps() -> None:
    """Populate matplotlib module globals. Idempotent."""
    global plt, mcolors
    if plt is not None:
        return
    import matplotlib
    matplotlib.use("Agg")  # headless safety
    import matplotlib.pyplot as _plt
    from matplotlib import colors as _mcolors
    _plt.style.use('seaborn-v0_8-whitegrid')
    _plt.rcParams['savefig.dpi'] = 300
    plt = _plt
    mcolors = _mcolors


# =============================================================================
# Input resolution & space detection
# =============================================================================

def _resolve_input(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    """Locate the input CSV from CLI arg, config, or error out."""
    path = (getattr(args, 'input', None)
            or (config.get('select_structures') or {}).get('input'))
    if not path:
        raise SystemExit(
            "select-structures: --input <CSV> is required (or set "
            "select_structures.input in config.json). Pass a pca_results.csv / "
            "tsne_results.csv / umap_results.csv produced by `cgkit analyze-atomic`."
        )
    if not os.path.exists(path):
        raise SystemExit(f"select-structures: input CSV not found: {path}")
    return path


def _detect_space(columns: List[str]) -> str:
    """Auto-detect 'pca' vs 'tsne' vs 'umap' from column names.

    Priority when multiple spaces are present: PCA > UMAP > t-SNE.
    """
    has_pc   = any(re.fullmatch(r'PC\d+',   c) for c in columns)
    has_umap = any(re.fullmatch(r'UMAP\d+', c) for c in columns)
    has_tsne = any(re.fullmatch(r'tSNE\d+', c) for c in columns)
    if has_pc:
        return 'pca'
    if has_umap:
        return 'umap'
    if has_tsne:
        return 'tsne'
    raise SystemExit(
        "select-structures: input CSV has none of PC<n>/tSNE<n>/UMAP<n> columns. "
        "Run `cgkit analyze-atomic` first to produce pca_results.csv / "
        "tsne_results.csv / umap_results.csv."
    )


def _feature_columns(columns: List[str], space: str) -> List[str]:
    """Return the ordered feature column names for the chosen space."""
    patterns = {
        'pca':   r'PC\d+',
        'tsne':  r'tSNE\d+',
        'umap':  r'UMAP\d+',
    }
    if space not in patterns:
        raise SystemExit(
            f"select-structures: unknown space {space!r} "
            f"(expected one of {sorted(patterns)})."
        )
    cols = [c for c in columns if re.fullmatch(patterns[space], c)]
    if not cols:
        raise SystemExit(
            f"select-structures: no {space.upper()} columns found in input CSV."
        )
    # Sort by trailing integer (PC2 before PC10).
    cols.sort(key=lambda c: int(re.search(r'\d+', c).group()))
    return cols


# =============================================================================
# Clustering
# =============================================================================

def _cluster(X: np.ndarray, method: str, n_clusters: int,
             eps: Optional[float], min_samples: int, seed: int) -> np.ndarray:
    """Cluster the feature matrix; return an int label array."""
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        k = max(1, min(n_clusters, len(X) // 2)) if len(X) >= 2 else 1
        if k < 1:
            k = 1
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        print(f"[cluster] KMeans k={k} on {len(X)} points")
        return labels
    elif method == 'dbscan':
        from sklearn.cluster import DBSCAN
        from scipy.spatial.distance import pdist
        if eps is None:
            if len(X) < 2:
                eps = 1.0
            else:
                eps = float(np.percentile(pdist(X), 30))
            print(f"[cluster] DBSCAN auto-eps (30th pct of pairwise dist) = {eps:.4f}")
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_noise = int(np.sum(labels == -1))
        n_clusters_real = len(set(labels) - {-1})
        print(f"[cluster] DBSCAN eps={eps:.4f} min_samples={min_samples}: "
              f"{n_clusters_real} clusters, {n_noise} noise points")
        return labels
    else:
        raise SystemExit(f"select-structures: unknown method {method!r}")


# =============================================================================
# Maximin (farthest-point) selection per cluster
# =============================================================================

def _select_per_cluster(X: np.ndarray, labels: np.ndarray, n: int,
                        include_noise: bool) -> Tuple[np.ndarray, Dict[int, int]]:
    """Greedy farthest-point sampling within each cluster.

    Returns:
        selected_mask : boolean array of len(X); True for selected points.
        rank_in_cluster : dict mapping global index -> selection rank within
            its cluster (0-based; 0 = the seed/centroid-nearest point).

    The first pick in each cluster is the point nearest the cluster centroid
    (deterministic, no RNG). Each subsequent pick maximises the minimum
    Euclidean distance to the already-chosen points — classic maximin / FPS.
    """
    selected_mask = np.zeros(len(X), dtype=bool)
    rank_in_cluster: Dict[int, int] = {}

    unique_labels = sorted(set(int(l) for l in labels))
    considered = [l for l in unique_labels if include_noise or l != -1]

    if not considered:
        raise SystemExit(
            "select-structures: DBSCAN labelled every point as noise (label -1). "
            "Re-run with --include-noise, or adjust --eps / --min-samples."
        )

    from scipy.spatial.distance import cdist

    for c in considered:
        idx_c = np.where(labels == c)[0]
        if len(idx_c) == 0:
            continue
        k = min(n, len(idx_c))
        Xc = X[idx_c]

        # Seed: point nearest the cluster centroid.
        centroid = Xc.mean(axis=0)
        seed_local = int(np.argmin(np.linalg.norm(Xc - centroid, axis=1)))
        chosen_local = [seed_local]

        while len(chosen_local) < k:
            # min distance from each candidate to the chosen set.
            d = cdist(Xc, Xc[chosen_local])          # (m, len(chosen))
            min_d = d.min(axis=1)
            # Exclude already-chosen by setting their score to -inf.
            min_d[chosen_local] = -np.inf
            nxt = int(np.argmax(min_d))
            chosen_local.append(nxt)

        for rank, local_i in enumerate(chosen_local):
            global_i = int(idx_c[local_i])
            selected_mask[global_i] = True
            rank_in_cluster[global_i] = rank

        if len(idx_c) < n:
            print(f"[small-cluster] cluster {c}: only {len(idx_c)} members "
                  f"(< N={n}), took all")

    return selected_mask, rank_in_cluster


# =============================================================================
# Frame extraction
# =============================================================================

def _parse_ts_from_structure_id(sid: str) -> Optional[int]:
    """Extract the timestep from a structure_id of form ``<sim>/<temp>@<ts>``."""
    if not isinstance(sid, str) or '@' not in sid:
        return None
    try:
        return int(sid.rsplit('@', 1)[-1])
    except (ValueError, IndexError):
        return None


def _sanitize(name: str) -> str:
    """Make a structure_id safe to use as a filename."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def _resolve_timestep(row: pd.Series) -> Optional[int]:
    """Get the timestep from the row, falling back to structure_id parsing."""
    ts = row.get('timestep')
    if ts is not None and not (isinstance(ts, float) and np.isnan(ts)):
        try:
            return int(ts)
        except (ValueError, TypeError):
            pass
    sid = row.get('structure_id')
    if sid is not None:
        return _parse_ts_from_structure_id(str(sid))
    return None


def _extract_and_copy(selected: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Copy each selected frame into its own dump file.

    Groups by ``source_file`` so each source dump is parsed at most once.
    Mutates/returns ``selected`` with added ``timestep`` (always populated,
    resolved from the column or parsed from ``structure_id``) and
    ``output_file`` columns (empty string for frames that could not be located).
    """
    selected = selected.copy()
    # Always have an explicit `timestep` column — either the source CSV's, or
    # parsed from structure_id. Manifest depends on it.
    selected['timestep'] = selected.apply(_resolve_timestep, axis=1)
    selected['output_file'] = ''
    ensure_dir(output_dir)

    skip_no_source = 0
    skip_no_frame = 0
    written = 0

    for source_file, group in selected.groupby('source_file', sort=False):
        if not isinstance(source_file, str) or not os.path.exists(source_file):
            skip_no_source += len(group)
            continue
        reader = LammpsDumpReader(source_file)
        if not reader.parse_file():
            skip_no_source += len(group)
            continue
        ts_to_idx = {ts: i for i, ts in enumerate(reader.timesteps)}

        for row_idx, row in group.iterrows():
            ts = row['timestep']
            if ts is None or (isinstance(ts, float) and np.isnan(ts)) \
               or int(ts) not in ts_to_idx:
                skip_no_frame += 1
                sid = row.get('structure_id', '?')
                print(f"[no-frame] {sid} @timestep={ts} not found in "
                      f"{os.path.basename(source_file)}; skipped")
                continue
            frame = reader.frames[ts_to_idx[int(ts)]]
            sid = str(row.get('structure_id', f"frame_{ts}"))
            out_name = _sanitize(sid) + '.lammpstrj'
            out_path = os.path.join(output_dir, out_name)
            write_lammps_frame(frame, out_path)
            selected.at[row_idx, 'output_file'] = out_name
            written += 1

    print(f"[extract] wrote {written} frames; "
          f"skipped {skip_no_source} (source missing), {skip_no_frame} (timestep not found)")
    return selected


# =============================================================================
# Manifest
# =============================================================================

def _write_manifest(selected: pd.DataFrame, feat_cols: List[str],
                    output_dir: str) -> str:
    """Write selection_manifest.csv with provenance columns."""
    base_cols = ['structure_id', 'cluster', 'selection_rank']
    optional_cols = [c for c in ['source_file', 'timestep', 'temp']
                     if c in selected.columns]
    out_cols = base_cols + optional_cols + feat_cols + ['output_file']
    # Only keep columns that exist.
    out_cols = [c for c in out_cols if c in selected.columns]
    manifest = selected[out_cols].copy()
    manifest_path = os.path.join(output_dir, 'selection_manifest.csv')
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved: {manifest_path}  ({len(manifest)} rows)")
    return manifest_path


# =============================================================================
# Overview figure
# =============================================================================

def _plot_selection_overview(df_clean: pd.DataFrame,
                             selected: pd.DataFrame,
                             feat_cols: List[str],
                             method: str,
                             output_dir: str) -> str:
    """Render ``selection_overview.png``: all points coloured by cluster with
    the N×K selected picks overlaid as prominent red markers.

    Best-effort: callers should wrap in try/except so a plotting failure can
    never block the (already-written) CSV / dump outputs. Returns the saved
    path, or ``''`` if the figure was skipped (e.g. <2 feature columns).
    """
    _import_deps()

    if len(feat_cols) < 2:
        print("[plot] fewer than 2 feature columns; skipping overview figure")
        return ''
    x_col, y_col = feat_cols[0], feat_cols[1]

    labels = df_clean['cluster'].values
    unique_labels = sorted(set(int(l) for l in labels))
    has_noise = -1 in unique_labels
    real_labels = [l for l in unique_labels if l != -1]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    # --- Background scatter: all points coloured by cluster ----------------
    if has_noise:
        noise_mask = (labels == -1)
        ax.scatter(df_clean.loc[noise_mask, x_col].values,
                   df_clean.loc[noise_mask, y_col].values,
                   c='#cccccc', s=10, alpha=0.4, label='Noise',
                   zorder=1)

    n_real = len(real_labels)
    if n_real > 0:
        cmap = plt.cm.get_cmap('tab10', 10)
        for i, c in enumerate(real_labels):
            mask = (labels == c)
            ax.scatter(df_clean.loc[mask, x_col].values,
                       df_clean.loc[mask, y_col].values,
                       c=[cmap(i % 10)], s=15, alpha=0.5,
                       label=f'Cluster {c}', zorder=2)

    # --- Foreground overlay: selected points ------------------------------
    # Match selected rows back to df_clean via structure_id (canonical key),
    # falling back to positional index alignment if not present.
    if 'structure_id' in selected.columns and 'structure_id' in df_clean.columns:
        sel_ids = set(selected['structure_id'].tolist())
        sel_mask = df_clean['structure_id'].isin(sel_ids).values
    else:
        sel_idx = selected.index.tolist()
        sel_mask = np.zeros(len(df_clean), dtype=bool)
        sel_mask[sel_idx] = True

    sx = df_clean.loc[sel_mask, x_col].values
    sy = df_clean.loc[sel_mask, y_col].values
    ax.scatter(sx, sy, marker='o', facecolors='none', edgecolors='red',
               s=90, linewidths=1.6, zorder=4,
               label=f'Selected (N={len(sx)})')

    # Annotate selection_rank — only when total ≤30 to avoid clutter.
    n_selected = len(selected)
    if n_selected <= 30 and 'selection_rank' in selected.columns:
        if 'structure_id' in selected.columns:
            rank_map = dict(zip(selected['structure_id'].tolist(),
                                selected['selection_rank'].tolist()))
        else:
            rank_map = dict(zip(selected.index.tolist(),
                                selected['selection_rank'].tolist()))
        for _, row in df_clean[sel_mask].iterrows():
            key = (row['structure_id']
                   if 'structure_id' in df_clean.columns else row.name)
            rank = rank_map.get(key)
            if rank is None:
                continue
            ax.annotate(str(int(rank)),
                        xy=(row[x_col], row[y_col]),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color='darkred', zorder=5)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    n_clusters_disp = n_real + (1 if has_noise else 0)
    plural = '' if n_clusters_disp == 1 else 's'
    ax.set_title(f'Structure selection overview '
                 f'({method}, {n_clusters_disp} cluster{plural}, '
                 f'{n_selected} selected)')

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8, labelspacing=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'selection_overview.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# =============================================================================
# Entry point
# =============================================================================

def run(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """cgkit select-structures entry point."""
    sel_cfg = config.get('select_structures', {}) or {}

    input_path = _resolve_input(args, config)
    output_dir = (getattr(args, 'output_dir', None)
                  or sel_cfg.get('output_dir')
                  or os.path.join(os.path.dirname(os.path.abspath(input_path)),
                                  'selected'))
    ensure_dir(output_dir)

    n = int(getattr(args, 'n'))
    if n < 1:
        raise SystemExit("select-structures: --n must be >= 1")

    method = getattr(args, 'method', None) or sel_cfg.get('method', 'kmeans')
    n_clusters = (getattr(args, 'n_clusters', None)
                  or sel_cfg.get('n_clusters', 8))
    eps = getattr(args, 'eps', None)
    if eps is None:
        eps = sel_cfg.get('eps')
    min_samples = (getattr(args, 'min_samples', None)
                   or sel_cfg.get('min_samples', 5))
    seed = getattr(args, 'seed', None) or sel_cfg.get('seed', 42)
    include_noise = bool(getattr(args, 'include_noise', False)
                         or sel_cfg.get('include_noise', False))

    print(f"[select-structures] input      = {input_path}")
    print(f"[select-structures] output_dir = {output_dir}")
    print(f"[select-structures] N per cluster = {n}")

    # 1. Load CSV.
    df = pd.read_csv(input_path)
    print(f"[select-structures] loaded {len(df)} rows from CSV")

    # 2. Detect / validate space.
    requested_space = getattr(args, 'space', None)
    space = requested_space or _detect_space(list(df.columns))
    feat_cols = _feature_columns(list(df.columns), space)
    print(f"[select-structures] space={space}, features={feat_cols}")

    # 3. Drop rows with NaN in feature columns.
    df_clean = df.dropna(subset=feat_cols).reset_index(drop=True)
    if len(df_clean) < len(df):
        print(f"[select-structures] dropped {len(df) - len(df_clean)} "
              f"rows with NaN features")
    if len(df_clean) == 0:
        print("[select-structures] no valid rows after NaN drop; aborting")
        return 1

    X = df_clean[feat_cols].values.astype(float)

    # 4. Cluster.
    labels = _cluster(X, method, int(n_clusters), eps, int(min_samples), int(seed))
    df_clean['cluster'] = labels

    # 5. Maximin selection per cluster.
    selected_mask, rank_map = _select_per_cluster(
        X, labels, n, include_noise=include_noise)
    selected = df_clean[selected_mask].copy()
    selected['selection_rank'] = [rank_map[i] for i in selected.index]
    print(f"[select-structures] selected {len(selected)} structures across "
          f"{selected['cluster'].nunique()} cluster(s)")

    if len(selected) == 0:
        print("[select-structures] nothing selected; aborting")
        return 1

    # 6. Extract & copy frames.
    selected = _extract_and_copy(selected, output_dir)

    # 7. Manifest.
    _write_manifest(selected, feat_cols, output_dir)

    # 8. Overview figure — best-effort, never fatal.
    try:
        _plot_selection_overview(df_clean, selected, feat_cols,
                                 method, output_dir)
    except Exception as e:
        print(f"[plot] selection_overview failed (non-fatal): {e}")

    return 0
