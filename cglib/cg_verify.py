"""cg-verify domain logic: validate CG CSV outputs against atomic source dumps.

Two modes (selected by ``--mode``):

* ``auto`` (default) — full PBC / conservation / coverage / manual-fidelity
  scan over one or many CG CSV files. Emits stdout + optional CSV report.
* ``manual`` — for a single CG CSV, locate which CG particle owns each
  user-supplied atom ID and report the particle's siblings / position /
  PBC status.

Design notes
------------
- Re-uses :class:`cglib.lammps.LammpsDumpReader` for atomic parsing and
  :func:`cglib.parallel.run_parallel` for multi-file scans (worker must
  return the 3-tuple ``(ok, msg, result)`` per the stable parallel contract).
- Assumes orthogonal simulation boxes (the parser drops tilt factors; a
  non-zero tilt triggers a one-line INFO warning but the scan continues).
- Conservation check **exactly mirrors** :func:`cglib.cg_gen.create_cg_particle`:
  position = center atom (first row of group); force / PE = mean when the
  corresponding ``average_*`` flag is set in the ``coarse_graining`` config.
- ``atom_indices`` (0-based row indices into the atomic DataFrame) is the
  canonical key — it is populated for **both** manual-assignment and
  pattern-based CG particles. ``assigned_atom_ids`` is only populated on
  manual rows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import parallel as _parallel
from .cg_gen import _unwrap_chain_coords
from .lammps import LammpsDumpReader
from .paths import ensure_dir, substitute_temp


# =============================================================================
# Type aliases
# =============================================================================

Issue = Dict[str, Any]
# task = (cg_csv, atomic_path, sim_name, temp, settings, cg_config)
Task = Tuple[str, Optional[str], str, Optional[int], Dict[str, Any], Dict[str, Any]]

_AXIS_NAMES = ('x', 'y', 'z')


# =============================================================================
# Settings resolution
# =============================================================================

def _resolve_settings(args: argparse.Namespace,
                      config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``verify_cg`` config with CLI overrides and the related
    ``paths`` / ``coarse_graining`` sections. Returns a flat dict."""
    verify_cfg = config.get('verify_cg', {}) or {}
    paths_cfg = config.get('paths', {}) or {}
    cg_cfg = config.get('coarse_graining', {}) or {}

    base_dir = (getattr(args, 'base_dir', None)
                or paths_cfg.get('cg_data_base_dir'))
    atomic_dir = (getattr(args, 'atomic_dir', None)
                  or paths_cfg.get('aa_data_base_dir')
                  or paths_cfg.get('base_dir'))

    output_dir = (getattr(args, 'output_dir', None)
                  or verify_cfg.get('output_dir')
                  or base_dir)

    checks = (getattr(args, 'checks', None)
              or verify_cfg.get('checks')
              or ['pbc', 'conservation', 'coverage', 'manual'])

    settings: Dict[str, Any] = {
        # Paths
        'base_dir': base_dir,
        'atomic_dir': atomic_dir,
        'output_dir': output_dir,
        # Mode + scope
        'mode': getattr(args, 'mode', 'auto') or 'auto',
        'atom_ids': list(getattr(args, 'atoms', None) or []),
        'scan_all': bool(getattr(args, 'all', False)),
        'max_files': getattr(args, 'max_files', None),
        'explicit_file': getattr(args, 'file', None),
        # Filters
        'sim_filter': list(getattr(args, 'sim', None) or []),
        'temp_filter': list(getattr(args, 'temp', None) or []),
        # Parallel
        'max_workers': getattr(args, 'workers', None),
        'parallel': False,  # default: serial for clean stdout interleaving
        # Output
        'no_csv': bool(getattr(args, 'no_csv', False)),
        'quiet': bool(getattr(args, 'quiet', False)),
        'failures_only': bool(getattr(args, 'failures_only', False)),
        # Tolerances
        'force_tol': (getattr(args, 'force_tol', None)
                      or verify_cfg.get('force_tolerance', 1.0e-4)),
        'pe_tol': (getattr(args, 'pe_tol', None)
                   or verify_cfg.get('pe_tolerance', 1.0e-6)),
        'pbc_thresh': (getattr(args, 'pbc_thresh', None)
                       or verify_cfg.get('pbc_span_threshold', 0.45)),
        # Active checks
        'checks': checks,
        # CG config (mirrored)
        'position_source': cg_cfg.get('position_source', 'unwrapped'),
        'average_forces': cg_cfg.get('average_forces', True),
        'average_potential_energy': cg_cfg.get('average_potential_energy', True),
        # New algorithm knobs (mirrored from coarse_graining)
        'unwrap_pbc': cg_cfg.get('unwrap_pbc', False),
        'r_cutoff': cg_cfg.get('r_cutoff', None),
        'id_patterns': cg_cfg.get('id_patterns', []) or [],
    }
    return settings


# =============================================================================
# File discovery + atomic path resolution
# =============================================================================

def _list_sim_candidates(settings: Dict[str, Any],
                         sim_configs: List[Dict[str, Any]]
                         ) -> List[Tuple[str, str, Optional[int]]]:
    """List (cg_csv, sim_name, temp) for every enabled sim whose name passes
    the --sim filter. Respects per-sim ``data_subdir`` / ``output_subdir``."""
    sim_filter = set(settings['sim_filter']) if settings['sim_filter'] else None
    out: List[Tuple[str, str, Optional[int]]] = []
    for sim_cfg in sim_configs:
        if not sim_cfg.get('enabled', True):
            continue
        name = sim_cfg['name']
        if sim_filter is not None and name not in sim_filter:
            continue
        # Try each temp; CG output lives under cg_data_base_dir/<output_subdir>
        temps = sim_cfg.get('temperatures') or [None]
        for temp in temps:
            sub = sim_cfg.get('output_subdir') or sim_cfg.get('data_subdir') or ''
            if temp is not None:
                sub = sub.replace('{temp}', str(temp))
            sub_dir = os.path.join(settings['base_dir'] or '', sub)
            csvs = sorted(glob.glob(os.path.join(sub_dir, '*_particles.csv')))
            for csv in csvs:
                out.append((csv, name, temp))
    return out


def _apply_temp_filter(candidates: List[Tuple[str, str, Optional[int]]],
                       temp_filter: List[int]) -> List[Tuple[str, str, Optional[int]]]:
    if not temp_filter:
        return candidates
    allow = set(temp_filter)
    return [c for c in candidates if c[2] is None or c[2] in allow]


def discover_tasks(settings: Dict[str, Any],
                   sim_configs: List[Dict[str, Any]]) -> List[Task]:
    """Build the list of (cg_csv, atomic_path, sim, temp, settings, cg_config)
    tasks to be verified."""
    cg_cfg = settings.get('_cg_config', {})  # injected by run()
    candidates = _list_sim_candidates(settings, sim_configs)
    candidates = _apply_temp_filter(candidates, settings['temp_filter'])
    candidates.sort()

    if settings['explicit_file']:
        # --file wins; sim/temp/all are ignored.
        cg_csv = os.path.abspath(settings['explicit_file'])
        atomic = _resolve_atomic_path(cg_csv, None, settings, None)
        return [(cg_csv, atomic, '(explicit)', None, settings, cg_cfg)]

    if not settings['scan_all']:
        # Default single-file mode: first candidate after sort.
        if not candidates:
            return []
        cg_csv, sim, temp = candidates[0]
        # Find the matching sim_cfg (for atomic subdir resolution)
        sim_cfg = next((s for s in sim_configs if s['name'] == sim), None)
        atomic = _resolve_atomic_path(cg_csv, sim_cfg, settings, temp)
        return [(cg_csv, atomic, sim, temp, settings, cg_cfg)]

    # --all mode: every candidate, optionally capped
    tasks: List[Task] = []
    for cg_csv, sim, temp in candidates:
        sim_cfg = next((s for s in sim_configs if s['name'] == sim), None)
        atomic = _resolve_atomic_path(cg_csv, sim_cfg, settings, temp)
        tasks.append((cg_csv, atomic, sim, temp, settings, cg_cfg))
        if settings['max_files'] and len(tasks) >= settings['max_files']:
            break
    return tasks


def _resolve_atomic_path(cg_csv: str,
                         sim_cfg: Optional[Dict[str, Any]],
                         settings: Dict[str, Any],
                         temp: Optional[int] = None) -> Optional[str]:
    """Given a CG ``*_particles.csv`` path, find its source atomic dump.

    The basename convention is::

        BASE.lammpstrj  ->  BASE.lammpstrj_particles.csv

    so we strip the ``_particles.csv`` suffix to recover ``BASE.lammpstrj``
    and search under ``atomic_dir`` (and, if available, inside the sim
    subdir). ``temp`` is substituted into any ``{temp}`` placeholder in the
    sim's subdir templates before searching, so layouts like
    ``<sim>/<temp>/traj/`` resolve correctly even when many sims share the
    same atomic dump filename (e.g. ``NPT2.300.100000`` exists once per
    pressure)."""
    if not cg_csv:
        return None
    fname = os.path.basename(cg_csv)
    if not fname.endswith('_particles.csv'):
        return None
    base_name = fname[:-len('_particles.csv')]  # e.g. foo.lammpstrj

    atomic_dir = settings['atomic_dir']
    if not atomic_dir or not os.path.isdir(atomic_dir):
        return None

    # Candidate search locations (in order of preference).
    search_roots: List[str] = []
    if sim_cfg:
        for key in ('trajectory_dir', 'data_subdir', 'output_subdir'):
            sub = sim_cfg.get(key)
            if sub:
                sub = substitute_temp(sub, temp)
                search_roots.append(os.path.join(atomic_dir, sub))
    search_roots.append(atomic_dir)

    for root in search_roots:
        candidate = os.path.join(root, base_name)
        if os.path.exists(candidate):
            return candidate

    # Recursive fallback (slow but robust to layout changes). NOTE: when
    # multiple files share the same basename (e.g. per-pressure NPT2.* dumps),
    # the recursive fallback is ambiguous — the sorted-first match may belong
    # to a different sim. Always prefer a sim-scoped path above.
    matches = sorted(glob.glob(os.path.join(atomic_dir, '**', base_name),
                               recursive=True))
    return matches[0] if matches else None


# =============================================================================
# IO
# =============================================================================

def _load_cg_csv(cg_csv: str) -> pd.DataFrame:
    """Load a particles CSV, coercing the list-valued text columns."""
    df = pd.read_csv(cg_csv)
    return df


def _atomic_df_for_timestep(reader: LammpsDumpReader,
                            ts: int) -> Optional[pd.DataFrame]:
    """Return the atomic DataFrame whose timestep matches ``ts``."""
    if ts not in reader.timesteps:
        return None
    idx = reader.timesteps.index(ts)
    return reader.get_dataframe(idx)


def _detect_triclinic(atomic_path: str) -> bool:
    """Quick scan of the first ``BOX BOUNDS`` block: returns True if any of
    the three box lines carries a non-zero tilt factor (third float)."""
    if not atomic_path or not os.path.exists(atomic_path):
        return False
    try:
        with open(atomic_path) as f:
            for line in f:
                if line.startswith('ITEM: BOX BOUNDS'):
                    for _ in range(3):
                        parts = f.readline().split()
                        if len(parts) >= 3:
                            try:
                                if abs(float(parts[2])) > 1.0e-12:
                                    return True
                            except ValueError:
                                pass
                    return False
    except OSError:
        return False
    return False


def _has_unwrapped(atomic_df: pd.DataFrame) -> bool:
    return all(col in atomic_df.columns for col in ('xu', 'yu', 'zu'))


def _coord_cols(atomic_df: pd.DataFrame, want_unwrapped: bool) -> Tuple[str, str, str]:
    if want_unwrapped and _has_unwrapped(atomic_df):
        return ('xu', 'yu', 'zu')
    return ('x', 'y', 'z')


def _prepare_atomic_df(atomic_df: pd.DataFrame,
                       box_bounds: List[List[float]],
                       settings: Dict[str, Any]) -> pd.DataFrame:
    """Return an atomic DataFrame with coords consistent with how cg-gen
    produced the CG CSV under audit.

    * If ``unwrap_pbc=True``: apply the same chain-style unwrap that cg-gen
      uses, so member atoms that span a periodic boundary in raw wrapped
      coords appear continuous (and the stored CG position, which was
      copied from the unwrapped center atom, can be reproduced exactly).
    * Otherwise: return the raw DataFrame unchanged.

    The function copies the input (only when mutating) so callers retain
    the original.
    """
    if not settings.get('unwrap_pbc', False):
        return atomic_df
    if atomic_df is None or len(atomic_df) == 0:
        return atomic_df
    return _unwrap_chain_coords(atomic_df.copy(), box_bounds)


def _pbc_abs_diff(a: float, b: float, box_bounds: List[List[float]],
                  axis: int) -> float:
    """Minimum-image absolute difference |a - b| under PBC along one axis."""
    L = box_bounds[axis][1] - box_bounds[axis][0]
    if abs(L) < 1.0e-12:
        return abs(a - b)
    d = a - b
    return abs(d - round(d / L) * L)


def _safe_json_int_list(raw: Any) -> List[int]:
    """Parse a CSV cell that should hold a JSON list of ints. Tolerates
    empty strings, NaN, and Python-style ``[1, 2, 3]`` text."""
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    s = str(raw).strip()
    if not s or s.lower() == 'nan':
        return []
    try:
        return [int(x) for x in json.loads(s)]
    except (json.JSONDecodeError, TypeError, ValueError):
        # Some cells store bare ``[1, 2, 3]`` text from older cg-gen runs;
        # ast.literal_eval handles those.
        try:
            import ast
            return [int(x) for x in ast.literal_eval(s)]
        except Exception:
            return []


def _bool_cell(raw: Any) -> bool:
    """Coerce a manual_assignment cell (True / NaN / '') to bool."""
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and np.isnan(raw):
            return False
        return bool(raw)
    s = str(raw).strip().lower()
    return s in ('true', '1', 'yes')


# =============================================================================
# Checks (auto mode)
# =============================================================================

def _issue(check: str, severity: str, cg_id: Optional[int], timestep: int,
           sim: str, file_basename: str, message: str,
           **details) -> Issue:
    return {
        'check': check, 'severity': severity, 'cg_id': cg_id,
        'timestep': timestep, 'sim': sim, 'file': file_basename,
        'message': message, 'details': details,
    }


def verify_pbc_span(cg_frame: pd.DataFrame,
                    atomic_df: pd.DataFrame,
                    box_bounds: List[List[float]],
                    sim: str, file_basename: str,
                    settings: Dict[str, Any]) -> List[Issue]:
    """Detect CG particles whose member atoms span a periodic boundary."""
    threshold = float(settings['pbc_thresh'])
    warn_threshold = 0.5 * threshold
    L = [box_bounds[i][1] - box_bounds[i][0] for i in range(3)]
    if any(abs(l) < 1.0e-12 for l in L):
        return []  # malformed box; skip silently
    # Apply chain unwrap if cg-gen was run with unwrap_pbc=True (otherwise a no-op)
    atomic_df = _prepare_atomic_df(atomic_df, box_bounds, settings)
    want_unwrapped = (settings['position_source'] == 'unwrapped')
    cols = _coord_cols(atomic_df, want_unwrapped)

    issues: List[Issue] = []
    ts = int(cg_frame['timestep'].iloc[0]) if 'timestep' in cg_frame else 0
    for _, row in cg_frame.iterrows():
        if _bool_cell(row.get('manual_assignment')):
            continue  # manual assignments may legitimately span a boundary
        idxs = _safe_json_int_list(row.get('atom_indices'))
        if len(idxs) < 2:
            continue
        try:
            member = atomic_df.iloc[idxs]
        except (IndexError, KeyError):
            continue  # surfaced by conservation check instead
        span = [float(member[c].max()) - float(member[c].min()) for c in cols]
        frac = [span[i] / L[i] for i in range(3)]
        worst = int(np.argmax(frac))
        if frac[worst] > threshold:
            issues.append(_issue(
                'pbc', 'FAIL', int(row['id']) if 'id' in row else None,
                ts, sim, file_basename,
                f"CG particle spans {frac[worst]*100:.1f}% of L_{_AXIS_NAMES[worst]} "
                f"(threshold {threshold*100:.0f}%)",
                span_frac=frac, axis=_AXIS_NAMES[worst],
                member_atom_ids=member['id'].astype(int).tolist()
                if 'id' in member else None,
            ))
        elif frac[worst] > warn_threshold:
            issues.append(_issue(
                'pbc', 'WARN', int(row['id']) if 'id' in row else None,
                ts, sim, file_basename,
                f"CG particle near PBC boundary: {frac[worst]*100:.1f}% of "
                f"L_{_AXIS_NAMES[worst]}",
                span_frac=frac, axis=_AXIS_NAMES[worst],
            ))
    return issues


def verify_conservation(cg_frame: pd.DataFrame,
                        atomic_df: pd.DataFrame,
                        box_bounds: List[List[float]],
                        cg_config: Dict[str, Any],
                        sim: str, file_basename: str,
                        settings: Dict[str, Any]) -> Tuple[List[Issue], Dict[str, float]]:
    """Recompute CG particle position/force/PE and compare against the stored
    values. Returns (issues, max_errors_dict)."""
    ftol = float(settings['force_tol'])
    etol = float(settings['pe_tol'])
    avg_F = bool(settings['average_forces'])
    avg_E = bool(settings['average_potential_energy'])
    want_unwrapped = (settings['position_source'] == 'unwrapped')
    # Apply chain unwrap if cg-gen was run with unwrap_pbc=True (otherwise a no-op)
    atomic_df = _prepare_atomic_df(atomic_df, box_bounds, settings)
    pos_cols = _coord_cols(atomic_df, want_unwrapped)

    issues: List[Issue] = []
    max_F = max_E = max_pos = 0.0
    ts = int(cg_frame['timestep'].iloc[0]) if 'timestep' in cg_frame else 0

    for _, row in cg_frame.iterrows():
        idxs = _safe_json_int_list(row.get('atom_indices'))
        if not idxs:
            issues.append(_issue(
                'conservation', 'FAIL',
                int(row['id']) if 'id' in row else None, ts, sim, file_basename,
                "empty atom_indices", ))
            continue
        try:
            member = atomic_df.iloc[idxs]
        except (IndexError, KeyError) as exc:
            issues.append(_issue(
                'conservation', 'FAIL',
                int(row['id']) if 'id' in row else None, ts, sim, file_basename,
                f"atom_indices out of range: {exc}",
                atom_indices=idxs,
            ))
            continue

        center = member.iloc[0]
        # Position (exact copy of center atom); compare under PBC minimum image
        # so wrapped vs unwrapped representations don't trigger false positives.
        exp_pos = [float(center[c]) for c in pos_cols]
        stored_pos = [float(row.get('x', np.nan)),
                      float(row.get('y', np.nan)),
                      float(row.get('z', np.nan))]
        pos_err = max(_pbc_abs_diff(exp_pos[i], stored_pos[i], box_bounds, i)
                      for i in range(3))

        # Force.
        if avg_F and all(c in member for c in ('fx', 'fy', 'fz')):
            exp_F = [float(member['fx'].mean()),
                     float(member['fy'].mean()),
                     float(member['fz'].mean())]
        else:
            exp_F = [float(center.get('fx', 0.0)),
                     float(center.get('fy', 0.0)),
                     float(center.get('fz', 0.0))]
        stored_F = [float(row.get('fx', np.nan)),
                    float(row.get('fy', np.nan)),
                    float(row.get('fz', np.nan))]
        F_err = max(abs(exp_F[i] - stored_F[i]) for i in range(3))

        # PE.
        E_err = 0.0
        if avg_E and 'c_pe' in member and 'c_pe' in row:
            exp_E = float(member['c_pe'].mean())
            stored_E = float(row['c_pe'])
            if not (np.isnan(exp_E) or np.isnan(stored_E)):
                E_err = abs(exp_E - stored_E)

        max_F = max(max_F, F_err)
        max_E = max(max_E, E_err)
        max_pos = max(max_pos, pos_err)

        if pos_err > 1.0e-9 or F_err > ftol or E_err > etol:
            issues.append(_issue(
                'conservation', 'FAIL',
                int(row['id']) if 'id' in row else None, ts, sim, file_basename,
                f"recompute mismatch: pos_err={pos_err:.2e} "
                f"F_err={F_err:.2e} E_err={E_err:.2e} "
                f"(tol: F={ftol:.0e}, E={etol:.0e})",
                pos_err=pos_err, force_err=F_err, pe_err=E_err,
                expected_force=exp_F, stored_force=stored_F,
                n_atoms=len(idxs),
            ))

    return issues, {'force_err': max_F, 'pe_err': max_E, 'pos_err': max_pos}


def verify_coverage(cg_frame: pd.DataFrame,
                    atomic_df: pd.DataFrame,
                    sim: str, file_basename: str,
                    settings: Dict[str, Any]) -> List[Issue]:
    """Every atomic row must appear in exactly one CG particle."""
    seen: List[int] = []
    for _, row in cg_frame.iterrows():
        seen.extend(_safe_json_int_list(row.get('atom_indices')))

    n_atoms = len(atomic_df)
    missing = sorted(set(range(n_atoms)) - set(seen))
    counts = Counter(seen)
    dups = sorted(i for i, c in counts.items() if c > 1)

    ts = int(cg_frame['timestep'].iloc[0]) if 'timestep' in cg_frame else 0
    issues: List[Issue] = []
    if missing:
        atom_ids_missing = (atomic_df.iloc[missing]['id'].astype(int).tolist()
                            if 'id' in atomic_df else missing)
        issues.append(_issue(
            'coverage', 'FAIL', None, ts, sim, file_basename,
            f"{len(missing)} atom(s) not in any CG particle",
            count=len(missing),
            missing_row_indices=missing[:50],
            missing_atom_ids=atom_ids_missing[:50],
        ))
    if dups:
        atom_ids_dups = (atomic_df.iloc[dups]['id'].astype(int).tolist()
                         if 'id' in atomic_df else dups)
        issues.append(_issue(
            'coverage', 'FAIL', None, ts, sim, file_basename,
            f"{len(dups)} atom(s) appear in >1 CG particle",
            count=len(dups),
            duplicate_row_indices=dups[:50],
            duplicate_atom_ids=atom_ids_dups[:50],
        ))
    if not missing and not dups and len(seen) != n_atoms:
        issues.append(_issue(
            'coverage', 'FAIL', None, ts, sim, file_basename,
            f"size mismatch: {len(seen)} assigned vs {n_atoms} atomic rows",
            count_delta=len(seen) - n_atoms,
        ))
    return issues


def verify_manual_fidelity(cg_frame: pd.DataFrame,
                           cg_config: Dict[str, Any],
                           sim: str, file_basename: str,
                           settings: Dict[str, Any]) -> List[Issue]:
    """Each entry in ``coarse_graining.cg_assignments`` must correspond to
    exactly one ``manual_assignment=True`` CG row with matching atom IDs
    and CG type."""
    assignments = cg_config.get('cg_assignments', []) or []
    if not assignments:
        return []

    expected: Dict[Tuple[int, ...], Dict[str, Any]] = {
        tuple(sorted(int(x) for x in a.get('atom_ids', []))): a
        for a in assignments
    }
    actual_rows = cg_frame[cg_frame['manual_assignment'].apply(_bool_cell)] \
        if 'manual_assignment' in cg_frame else cg_frame.iloc[0:0]
    actual: Dict[Tuple[int, ...], List[pd.Series]] = {}
    for _, row in actual_rows.iterrows():
        ids = tuple(sorted(_safe_json_int_list(row.get('assigned_atom_ids'))))
        actual.setdefault(ids, []).append(row)

    ts = int(cg_frame['timestep'].iloc[0]) if 'timestep' in cg_frame else 0
    issues: List[Issue] = []
    for ids, a in expected.items():
        if not ids:
            continue
        rows = actual.get(ids, [])
        if not rows:
            issues.append(_issue(
                'manual', 'FAIL', None, ts, sim, file_basename,
                f"manual assignment {list(ids)} missing from CG CSV",
                expected_atom_ids=list(ids),
                expected_cg_type=a.get('cg_type'),
            ))
            continue
        if len(rows) > 1:
            issues.append(_issue(
                'manual', 'FAIL', None, ts, sim, file_basename,
                f"manual assignment {list(ids)} duplicated {len(rows)} times",
                cg_ids=[int(r['id']) for r in rows if 'id' in r],
            ))
            continue
        row = rows[0]
        expected_type = a.get('cg_type')
        if expected_type is not None and 'type' in row \
                and int(row['type']) != int(expected_type):
            issues.append(_issue(
                'manual', 'FAIL', int(row['id']) if 'id' in row else None,
                ts, sim, file_basename,
                f"CG type mismatch for {list(ids)}: "
                f"expected {expected_type}, got {row['type']}",
                expected_cg_type=expected_type, stored_cg_type=int(row['type']),
            ))

    # Reverse: undeclared manual rows.
    for ids, rows in actual.items():
        if not ids:
            continue
        if ids not in expected:
            issues.append(_issue(
                'manual', 'WARN', None, ts, sim, file_basename,
                f"undeclared manual assignment in CSV: {list(ids)}",
                cg_ids=[int(r['id']) for r in rows if 'id' in r],
            ))
    return issues


# =============================================================================
# Per-file worker (picklable; called by run_parallel)
# =============================================================================

def _process_one_case(task: Task) -> Tuple[bool, str, Dict[str, Any]]:
    """Verify one CG CSV against its source atomic dump.

    Returns the 3-tuple ``(ok, msg, result)`` per the run_parallel contract.
    ``ok=False`` indicates a *file-level* error (missing / unparseable /
    NaN-bearing / timestep-mismatched); per-particle issues are reported
    through ``result['issues']`` regardless of ``ok``."""
    cg_csv, atomic_path, sim, temp, settings, cg_config = task
    file_basename = os.path.basename(cg_csv)

    if not atomic_path or not os.path.exists(atomic_path):
        return False, f"atomic source not found for {file_basename}", {}
    if not os.path.exists(cg_csv):
        return False, f"CG CSV not found: {cg_csv}", {}

    try:
        cg_df = _load_cg_csv(cg_csv)
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to read CG CSV {file_basename}: {exc!r}", {}

    # Quick NaN scan on the core numeric columns.
    core = ['id', 'type', 'x', 'y', 'z', 'fx', 'fy', 'fz',
            'n_atoms', 'atom_indices', 'timestep']
    present_core = [c for c in core if c in cg_df.columns]
    bad_cols = [c for c in present_core if cg_df[c].isna().any()]
    if bad_cols:
        return False, (f"NaN in CG CSV columns {bad_cols} of {file_basename}; "
                       "regenerate the CG data before verifying"), {}

    reader = LammpsDumpReader(atomic_path)
    if not reader.parse_file():
        return False, f"failed to parse atomic dump {atomic_path}", {}

    cg_timesteps = sorted(int(t) for t in cg_df['timestep'].unique())
    atomic_ts_set = set(reader.timesteps)
    if not set(cg_timesteps).issubset(atomic_ts_set):
        missing_ts = sorted(set(cg_timesteps) - atomic_ts_set)
        return False, (f"{file_basename}: CG timesteps {missing_ts[:5]} "
                       "not present in atomic file"), {}

    checks = settings['checks']
    issues: List[Issue] = []
    max_errs: Dict[str, float] = {'force_err': 0.0, 'pe_err': 0.0, 'pos_err': 0.0}
    per_frame_summary: List[Dict[str, Any]] = []
    triclinic = _detect_triclinic(atomic_path)

    for ts in cg_timesteps:
        cg_frame = cg_df[cg_df['timestep'] == ts]
        atomic_df = _atomic_df_for_timestep(reader, ts)
        if atomic_df is None:
            return False, f"atomic frame for t={ts} missing in {file_basename}", {}
        box = reader.box_bounds[reader.timesteps.index(ts)]

        frame_issues: List[Issue] = []
        if 'pbc' in checks:
            frame_issues += verify_pbc_span(cg_frame, atomic_df, box,
                                            sim, file_basename, settings)
        if 'conservation' in checks:
            fi, ferrs = verify_conservation(cg_frame, atomic_df, box, cg_config,
                                            sim, file_basename, settings)
            frame_issues += fi
            for k, v in ferrs.items():
                max_errs[k] = max(max_errs.get(k, 0.0), v)
        if 'coverage' in checks:
            frame_issues += verify_coverage(cg_frame, atomic_df,
                                            sim, file_basename, settings)
        if 'manual' in checks:
            frame_issues += verify_manual_fidelity(cg_frame, cg_config,
                                                   sim, file_basename, settings)

        n_particles = len(cg_frame)
        n_covered = sum(len(_safe_json_int_list(r.get('atom_indices')))
                        for _, r in cg_frame.iterrows())
        per_frame_summary.append({
            'timestep': ts,
            'n_particles': n_particles,
            'n_atoms_covered': n_covered,
            'n_atomic_atoms': len(atomic_df),
            'issues_by_check': Counter(i['check'] for i in frame_issues),
            'issues_by_severity': Counter(i['severity'] for i in frame_issues),
        })
        issues.extend(frame_issues)

    return True, '', {
        'file': file_basename,
        'atomic_file': os.path.basename(atomic_path),
        'sim': sim,
        'temp': temp,
        'n_frames': len(cg_timesteps),
        'n_atomic_atoms': int(reader.natoms[0]) if reader.natoms else 0,
        'triclinic': triclinic,
        'issues': issues,
        'max_errors': max_errs,
        'per_frame': per_frame_summary,
    }


# =============================================================================
# Manual mode
# =============================================================================

def _run_manual_lookup(task: Task, atom_ids: List[int]) -> int:
    """For each user-supplied atom ID, locate the owning CG particle and print
    a structured report."""
    cg_csv, atomic_path, sim, temp, settings, cg_config = task
    cg_df = _load_cg_csv(cg_csv)
    if atomic_path is None or not os.path.exists(atomic_path):
        print(f"[cg-verify] atomic source not found; cannot resolve row indices.")
        return 2

    reader = LammpsDumpReader(atomic_path)
    if not reader.parse_file():
        print(f"[cg-verify] failed to parse atomic dump: {atomic_path}")
        return 2

    print("\n" + "=" * 70)
    print("MANUAL LOOKUP")
    print("=" * 70)
    print(f"CG file       : {os.path.basename(cg_csv)}")
    print(f"Atomic source : {os.path.basename(atomic_path)}")
    print(f"Sim           : {sim}" + (f"  temp={temp}K" if temp is not None else ''))
    n_atomic = reader.natoms[0] if reader.natoms else 0
    print(f"Atomic frames : {len(reader.timesteps)} | atoms/frame: {n_atomic}")
    print(f"CG rows total : {len(cg_df)} | unique timesteps: "
          f"{cg_df['timestep'].nunique() if 'timestep' in cg_df else 1}")
    print()

    # ID → row-index map (atomic file may not be ID-sorted).
    atomic_first = reader.get_dataframe(0)
    if atomic_first is None:
        print("[cg-verify] atomic file has no frames.")
        return 2
    if 'id' not in atomic_first:
        print("[cg-verify] atomic dump has no 'id' column; cannot map IDs.")
        return 2
    id_to_idx = {int(aid): i for i, aid in enumerate(atomic_first['id'])}

    box0 = reader.box_bounds[0] if reader.box_bounds else None
    L = ([box0[i][1] - box0[i][0] for i in range(3)]
         if box0 else [0.0, 0.0, 0.0])
    # Mirror cg-gen's chain unwrap so PBC span reporting matches auto mode
    atomic_first = _prepare_atomic_df(
        atomic_first, box0 if box0 else [[0.0, 0.0]] * 3, settings)
    want_unwrapped = (settings['position_source'] == 'unwrapped') \
        and _has_unwrapped(atomic_first)
    coord_cols = _coord_cols(atomic_first, want_unwrapped)
    pbc_thresh = float(settings['pbc_thresh'])

    multi_hit_total = 0
    for aid in atom_ids:
        print(f"--- Atom ID {aid} ---")
        if aid not in id_to_idx:
            print(f"  NOT FOUND in atomic file (atomic IDs range "
                  f"{min(id_to_idx)}–{max(id_to_idx) if id_to_idx else 0}).")
            continue
        row_idx = id_to_idx[aid]
        print(f"  row index in atomic file : {row_idx}")

        hits: List[Tuple[int, pd.Series]] = []  # (cg_df index, row)
        for df_idx, cg_row in cg_df.iterrows():
            if row_idx in _safe_json_int_list(cg_row.get('atom_indices')):
                hits.append((df_idx, cg_row))
        if not hits:
            print(f"  ❌ NOT in any CG particle (possible coverage bug).")
            continue

        # Group hits by timestep; pick first hit per timestep for the report.
        by_ts: Dict[int, List[pd.Series]] = {}
        for _, r in hits:
            ts = int(r['timestep']) if 'timestep' in r else 0
            by_ts.setdefault(ts, []).append(r)

        for ts in sorted(by_ts):
            rows_this_ts = by_ts[ts]
            row = rows_this_ts[0]
            idxs = _safe_json_int_list(row.get('atom_indices'))
            atomic_for_ts = _atomic_df_for_timestep(reader, ts)
            if atomic_for_ts is None:
                atomic_for_ts = atomic_first
            try:
                member = atomic_for_ts.iloc[idxs]
                sib_ids = member['id'].astype(int).tolist() \
                    if 'id' in member else idxs
            except (IndexError, KeyError):
                sib_ids = idxs

            manual = _bool_cell(row.get('manual_assignment'))
            print(f"  @t={ts}:")
            print(f"    CG id        : {int(row['id']) if 'id' in row else '?'}")
            print(f"    CG type      : {int(row['type']) if 'type' in row else '?'}")
            print(f"    n_atoms      : {int(row['n_atoms']) if 'n_atoms' in row else '?'}")
            print(f"    pattern      : {row['pattern'] if 'pattern' in row and pd.notna(row['pattern']) else '(manual)'}")
            print(f"    manual       : {manual}")
            print(f"    member IDs   : {sib_ids}")

            # Position
            if all(c in row for c in ('x', 'y', 'z')):
                print(f"    CG pos       : ({row['x']:.4f}, {row['y']:.4f}, {row['z']:.4f})")
            src_atom = atomic_for_ts.iloc[row_idx]
            src_pos = [float(src_atom[c]) for c in coord_cols]
            print(f"    atomic pos   : ({src_pos[0]:.4f}, {src_pos[1]:.4f}, {src_pos[2]:.4f})"
                  f"  [{'unwrapped' if want_unwrapped else 'wrapped'}]")

            # Quick PBC-span check.
            if not manual and len(idxs) >= 2 and all(abs(l) > 1e-12 for l in L):
                try:
                    spans = [float(member[c].max()) - float(member[c].min())
                             for c in coord_cols]
                    fracs = [spans[i] / L[i] for i in range(3)]
                    worst = int(np.argmax(fracs))
                    if fracs[worst] > pbc_thresh:
                        print(f"    ⚠ PBC span   : {fracs[worst]*100:.1f}% of "
                              f"L_{_AXIS_NAMES[worst]} (>{pbc_thresh*100:.0f}%)")
                except Exception:
                    pass

            if len(rows_this_ts) > 1:
                multi_hit_total += len(rows_this_ts) - 1
                print(f"    ⚠ also appears in {len(rows_this_ts)-1} other CG "
                      f"particle(s) at this timestep — likely coverage bug.")
        print()

    if multi_hit_total:
        print(f"[cg-verify] {multi_hit_total} extra membership(s) detected; "
              "rerun without --mode manual to get the full FAIL report.")
        return 1
    return 0


# =============================================================================
# Reporting
# =============================================================================

def _aggregate_results(results: List[Tuple[Task, bool, str, Dict[str, Any]]]
                       ) -> Dict[str, Any]:
    """Build summary aggregates from the per-file results."""
    agg = {
        'n_files': len(results),
        'n_ok_files': 0,
        'n_bad_files': 0,         # file-level errors
        'n_failed_files': 0,      # validation FAILs (file loaded OK)
        'total_issues': 0,
        'total_fails': 0,
        'total_warns': 0,
        'worst_force_err': 0.0,
        'worst_pe_err': 0.0,
        'worst_pos_err': 0.0,
        'per_file': [],
        'all_issues': [],
        'triclinic_seen': False,
    }
    for task, ok, msg, result in results:
        if not ok:
            agg['n_bad_files'] += 1
            agg['per_file'].append({
                'file': os.path.basename(task[0]),
                'sim': task[2], 'temp': task[3],
                'status': 'BAD', 'error': msg,
                'n_frames': 0, 'issues': [],
                'max_errors': {},
            })
            continue
        agg['n_ok_files'] += 1
        issues = result.get('issues', []) or []
        fails = sum(1 for i in issues if i['severity'] == 'FAIL')
        warns = sum(1 for i in issues if i['severity'] == 'WARN')
        if fails > 0:
            agg['n_failed_files'] += 1
        agg['total_issues'] += len(issues)
        agg['total_fails'] += fails
        agg['total_warns'] += warns
        max_errs = result.get('max_errors', {}) or {}
        agg['worst_force_err'] = max(agg['worst_force_err'],
                                     float(max_errs.get('force_err', 0.0)))
        agg['worst_pe_err'] = max(agg['worst_pe_err'],
                                  float(max_errs.get('pe_err', 0.0)))
        agg['worst_pos_err'] = max(agg['worst_pos_err'],
                                   float(max_errs.get('pos_err', 0.0)))
        if result.get('triclinic'):
            agg['triclinic_seen'] = True
        agg['per_file'].append({
            'file': result['file'],
            'atomic_file': result.get('atomic_file', ''),
            'sim': result['sim'],
            'temp': result.get('temp'),
            'status': 'OK' if fails == 0 else 'FAIL',
            'n_frames': result.get('n_frames', 0),
            'n_atomic_atoms': result.get('n_atomic_atoms', 0),
            'per_frame': result.get('per_frame', []),
            'issues': issues,
            'max_errors': max_errs,
        })
        agg['all_issues'].extend(issues)
    return agg


def _print_stdout_report(agg: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """Pretty-print the auto-mode aggregate report."""
    bar = '=' * 70
    print()
    print(bar)
    print("CG-VERIFY: 校核 CG 输出与源原子 dump 的一致性")
    print(bar)
    print(f"CG 目录      : {settings['base_dir'] or '(?)'}")
    print(f"原子目录     : {settings['atomic_dir'] or '(?)'}")
    print(f"模式         : auto")
    print(f"扫描文件数   : {agg['n_files']}")
    print(f"检查项       : {', '.join(settings['checks'])}")
    if agg['triclinic_seen']:
        print("[INFO] 检测到三斜盒 (tilt 非零)；PBC 检查按正交盒近似处理")
    print()

    quiet = settings['quiet']
    for pf in agg['per_file']:
        if pf['status'] == 'BAD':
            print(f"[BAD] {pf['file']}")
            print(f"      {pf['error']}")
            continue
        if not quiet or pf['status'] == 'FAIL':
            print(f"------ {pf['file']} "
                  f"({pf.get('sim', '?')}"
                  + (f", temp={pf['temp']}K" if pf.get('temp') is not None else '')
                  + ") ------")
            print(f"  原子源 : {pf.get('atomic_file', '?')} "
                  f"({pf.get('n_atomic_atoms', 0)} atoms, {pf.get('n_frames', 0)} frames)")
            for fr in pf.get('per_frame', []):
                ts = fr['timestep']
                cov = fr['n_atoms_covered']
                nat = fr['n_atomic_atoms']
                ibc = fr['issues_by_check']
                ibs = fr['issues_by_severity']
                line = (f"    t={ts}: {fr['n_particles']} CG 粒子, "
                        f"覆盖 {cov}/{nat} 原子")
                if ibs:
                    parts = [f"{ibs[c]} {c}" for c in ('FAIL', 'WARN', 'INFO')
                             if ibs.get(c, 0)]
                    line += "  | " + ", ".join(parts)
                else:
                    line += "  | 全部 OK"
                print(line)
            max_errs = pf.get('max_errors', {}) or {}
            if max_errs:
                print(f"  最大误差 : F={max_errs.get('force_err', 0):.2e}, "
                      f"E={max_errs.get('pe_err', 0):.2e}, "
                      f"pos={max_errs.get('pos_err', 0):.2e}")
            # Show first few FAIL messages.
            fails = [i for i in pf['issues'] if i['severity'] == 'FAIL']
            for fi in fails[:5]:
                cg = f" CG#{fi['cg_id']}" if fi.get('cg_id') is not None else ''
                print(f"  [FAIL/{fi['check']}] t={fi['timestep']}{cg}: {fi['message']}")
            if len(fails) > 5:
                hint = '' if settings['no_csv'] else '; see CSV report'
                print(f"  ... ({len(fails) - 5} more FAILs{hint})")
            print()

    print(bar)
    print("SUMMARY")
    print(bar)
    print(f"  Files OK    : {agg['n_ok_files'] - agg['n_failed_files']}")
    print(f"  Files FAIL   : {agg['n_failed_files']}")
    print(f"  Files BAD    : {agg['n_bad_files']} (file-level errors)")
    print(f"  Total FAILs  : {agg['total_fails']}")
    print(f"  Total WARNs  : {agg['total_warns']}")
    print(f"  Worst errors : F={agg['worst_force_err']:.2e}, "
          f"E={agg['worst_pe_err']:.2e}, pos={agg['worst_pos_err']:.2e}")


def _write_csv_report(issues: List[Issue], path: str,
                      failures_only: bool) -> str:
    """Write a flat CSV of issues. Returns the path written, or '' if no
    rows and the file was not created."""
    rows = []
    for it in issues:
        if failures_only and it['severity'] != 'FAIL':
            continue
        det = it.get('details', {}) or {}
        span = det.get('span_frac') or [None, None, None]
        rows.append({
            'file': it['file'], 'sim': it['sim'], 'temp': None,
            'timestep': it['timestep'], 'check': it['check'],
            'severity': it['severity'], 'cg_id': it.get('cg_id'),
            'message': it['message'],
            'n_atoms': det.get('n_atoms'),
            'member_atom_ids': (';'.join(str(x) for x in det['member_atom_ids'])
                               if det.get('member_atom_ids') else ''),
            'force_err': det.get('force_err'),
            'pe_err': det.get('pe_err'),
            'pos_err': det.get('pos_err'),
            'pbc_span_frac_x': span[0] if span and span[0] is not None else None,
            'pbc_span_frac_y': span[1] if len(span) > 1 and span[1] is not None else None,
            'pbc_span_frac_z': span[2] if len(span) > 2 and span[2] is not None else None,
        })
    if not rows:
        return ''
    df = pd.DataFrame(rows, columns=[
        'file', 'sim', 'temp', 'timestep', 'check', 'severity', 'cg_id',
        'message', 'n_atoms', 'member_atom_ids',
        'force_err', 'pe_err', 'pos_err',
        'pbc_span_frac_x', 'pbc_span_frac_y', 'pbc_span_frac_z',
    ])
    ensure_dir(os.path.dirname(os.path.abspath(path)) or '.')
    df.to_csv(path, index=False)
    return path


# =============================================================================
# Entry point
# =============================================================================

def run(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """cgkit cg-verify entry point."""
    settings = _resolve_settings(args, config)
    settings['_cg_config'] = config.get('coarse_graining', {}) or {}

    # --- CLI sanity ------------------------------------------------------
    mode = settings['mode']
    if mode == 'manual' and not settings['atom_ids']:
        print("[cg-verify] --mode manual requires --atoms ID1 ID2 ...")
        return 3
    if mode == 'auto' and settings['atom_ids']:
        print("[cg-verify] --atoms is only valid with --mode manual")
        return 3
    if not settings['base_dir'] and not settings['explicit_file']:
        print("[cg-verify] no paths.cg_data_base_dir in config and no --file; "
              "nothing to verify.")
        return 3

    # --- Discover tasks --------------------------------------------------
    sim_configs = config.get('simulations', []) or []
    tasks = discover_tasks(settings, sim_configs)
    if not tasks:
        print("[cg-verify] no CG CSV files matched the current filters.")
        return 2

    selected_file = (tasks[0][0] if (len(tasks) == 1 or settings['explicit_file'])
                     else None)
    if len(tasks) == 1 and not settings['explicit_file']:
        print(f"[cg-verify] single-file mode: {os.path.basename(tasks[0][0])}")

    # --- Manual mode -----------------------------------------------------
    if mode == 'manual':
        # Force a unique file.
        unique_csvs = sorted({t[0] for t in tasks})
        if len(unique_csvs) > 1:
            print(f"[cg-verify] manual mode needs a unique CG CSV; "
                  f"{len(unique_csvs)} matched. Pass --file PATH.")
            return 3
        return _run_manual_lookup(tasks[0], settings['atom_ids'])

    # --- Auto mode -------------------------------------------------------
    if settings['explicit_file'] is None:
        # Show which file we picked in default single-file mode.
        cg_csv = tasks[0][0]
        atomic = tasks[0][1]
        print(f"[cg-verify] auto mode: {os.path.basename(cg_csv)}"
              + (f"  ↔  {os.path.basename(atomic)}" if atomic else "  (atomic not found)"))

    results = _parallel.run_parallel(
        tasks, _process_one_case,
        n_workers=settings['max_workers'],
        parallel=settings['parallel'],
        desc="  cg-verify", unit="file",
    )

    agg = _aggregate_results(results)
    _print_stdout_report(agg, settings)

    # CSV report
    csv_path = ''
    if not settings['no_csv']:
        out_dir = settings['output_dir'] or '.'
        target = os.path.join(out_dir, 'cg_verify_report.csv')
        csv_path = _write_csv_report(agg['all_issues'], target,
                                     settings['failures_only'])
    if csv_path:
        print(f"Report CSV : {csv_path}")
    elif not settings['no_csv']:
        print("Report CSV : (no issues — nothing to write)")

    # Exit code
    if agg['n_bad_files'] > 0 and agg['n_ok_files'] == 0:
        return 2  # all files were file-level errors
    if agg['n_bad_files'] > 0:
        return 2
    if agg['total_fails'] > 0:
        return 1
    return 0
