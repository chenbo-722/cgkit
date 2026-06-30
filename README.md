# cgkit — CH_CG Coarse-Graining Toolkit

`cgkit` is the single entry point for the **CH_CG cgkit** suite: a set of
Python utilities that take atomic-level LAMMPS trajectories of polyethylene (PE)
and produce coarse-grained (CG) training data for DeepMD-kit, plus statistical
and structural analyses of those CG datasets.

The five original standalone scripts (`02-get_CGdata_parall.py`,
`03-trans_CGnpy_parall.py`, `generate_fparam.py`, `0x-analyze_cg_data.py`,
`0x-analyze_atomic_structure.py`, ~5,200 lines combined) have been consolidated
into one executable + one importable library (`cglib/`) while preserving their
domain algorithms 1:1. The originals are archived under `legacy/` for reference.

| Surface      | Path                  | Purpose                                   |
|--------------|-----------------------|-------------------------------------------|
| Executable   | `cgkit` (entry point) | argparse dispatcher (≈ 80 lines)          |
| Library      | `cglib/`              | 11 importable modules                     |
| Config       | `config.json`         | unified config (8 top-level sections)     |
| Reference    | `legacy/`             | the five original scripts, frozen         |

---

## 1. Installation

### Install the `cgkit` command (recommended)

From the `cgkit/` project directory:

```bash
pip install -e .
```

This creates a `cgkit` executable on your `PATH` (in the conda/venv `bin/`).
You can then run any subcommand from **any directory**:

```bash
cgkit cg-gen
cgkit plot-pt --max-frames 200
```

`config.json` is auto-located relative to the source tree, so it is found
regardless of your current directory. The install is **editable** — edits to
`cgkit.py` / `cglib/` take effect immediately, no reinstall needed.

Install optional dependency groups for the subcommands you use:

```bash
pip install -e ".[atomic,soap]"     # analyze-atomic with SOAP descriptors
pip install -e ".[all]"             # everything (heavy: pulls torch, dscribe, …)
```

### Manual alternative (no install)

You can also run `cgkit.py` directly without installing — but only from the
project directory, since `cglib` must be importable:

```bash
cd cgkit && python cgkit.py cg-gen
```

### Required packages (all subcommands)

```bash
pip install numpy pandas tqdm
```

### Optional (per-subcommand dependency matrix)

| Subcommand           | Extra packages required                                                  |
|----------------------|--------------------------------------------------------------------------|
| `cg-gen`             | _(none beyond required)_                                                 |
| `to-deepmd`          | _(none beyond required)_                                                 |
| `fparam extract`     | _(none beyond required)_                                                 |
| `fparam const`       | _(none beyond required)_                                                 |
| `cg-verify`          | _(none beyond required)_                                                 |
| `analyze-cg`         | `matplotlib`, `scipy`                                                    |
| `analyze-atomic`     | `matplotlib`, `scipy`, `scikit-learn`, `networkx` (optional)             |
| `analyze-atomic` (SOAP)  | + `ase`, `dscribe`                                                   |
| `analyze-atomic` (GNN)   | + `torch`, `torch-geometric` (falls back to random embeddings)       |
| `analyze-atomic` (UMAP)  | + `umap-learn` (falls back to skipping UMAP step)                    |

Heavy packages are **lazy-loaded**: `import cglib` and any non-analysis
subcommand will succeed with just `numpy/pandas/tqdm` installed. The
`analyze-cg` / `analyze-atomic` subcommands import their extras on first use.

---

## 2. Quick start

```bash
# 1. Build CG CSVs from atomic LAMMPS dumps
cgkit cg-gen

# 2. Convert CG CSVs to DeepMD-kit .raw / .npy
cgkit to-deepmd

# 3. Generate fparam (per-frame T(t) from log.lammps)
cgkit fparam extract
cgkit fparam const                # constant-T frames for 1-npt/2-nvt

# 4. Statistical analysis of the CG dataset
cgkit analyze-cg

# 5. SOAP / PCA / t-SNE / clustering of CG trajectories
cgkit analyze-atomic --mode cg
```

Override anything on the CLI:

```bash
cgkit cg-gen --sim 1-npt --temp 200 300 --workers 4
cgkit fparam const --unit K       # Kelvin instead of eV
cgkit analyze-atomic --mode aa --max-frames 200
```

---

## 3. Subcommand reference

### `cgkit cg-gen`  *(legacy `02-get_CGdata_parall.py`)*

Reads LAMMPS dumps (`*.lammpstrj`) from `paths.base_dir/<sim>/traj/`, performs
flexible-pattern coarse-graining (e.g. `[1,2,2]` = 1 C + 2 H), and writes CG
particle / box CSV files plus optional CG dump trajectories.

**Config sections read:** `paths.base_dir`, `paths.cg_data_base_dir`,
`coarse_graining`, `processing`, `output`, `simulations`.

**CLI:**
```
--base-dir DIR       override paths.base_dir        (atomic input)
--output-dir DIR     override paths.cg_data_base_dir (CG CSV output)
--sim NAME [NAME...] filter simulations
--temp K [K...]      override temperatures
--workers N          parallel workers
--r-cutoff Å         pattern-matching cutoff radius (default: from config, 1.25)
--unwrap-pbc         enable chain-style PBC unwrap before CG matching
--no-unwrap-pbc      disable PBC unwrap (use raw wrapped coords)
```

**PBC unwrap behavior:** when `coarse_graining.unwrap_pbc=true` (default in
shipped configs), each atomic frame is run through `_unwrap_chain_coords`
— a by-id-order minimum-image fold that reconstructs physical coordinates
across periodic boundaries before pattern matching. This eliminates the
classic "wrapped-coordinate artifact picks two atoms on opposite sides of
the box" failure mode. The unwrap assumes consecutive atom ids are
chemically bonded (true for bonded dumps ordered by molecule); dumps that
list all C first then all H may not benefit. Override per-run with
`--no-unwrap-pbc`.

**`r_cutoff` distance cutoff:** when set (default `1.25` Å), type-2
candidates outside `r_cutoff` from a center atom are dropped before
nearest-neighbor selection. This prevents a center from "stealing" a
distant H that rightfully belongs to a nearer center. Set to `null` for
unlimited range (pre-upgrade behavior).

**`id_patterns` (id-offset binding):** optional intermediate layer between
`cg_assignments` (per-id manual) and `patterns` (per-distance automatic).
Each rule is `{type_pattern, id_offsets, cg_type}` where `id_offsets[i]`
is the signed offset from the center atom's id to the i-th member atom
(`id_offsets[0]` must be 0). Priority order: **cg_assignments →
id_patterns → patterns**. Useful for dumps with a uniform id layout
(e.g. `center_id+1, center_id+2` for the two H on each CH₂).

**Per-trajectory outputs:** `<basename>_particles.csv`,
`<basename>_box_vectors.csv`, optionally `<basename>_cg.lammpstrj`. The
particles CSV gains a `match_status` column (`manual` / `id_pattern` /
`pattern`) recording which stage produced each CG bead, plus an
`id_pattern` column with the matched rule when applicable.

---

### `cgkit to-deepmd`  *(legacy `03-trans_CGnpy_parall.py`)*

Converts CG particle/box CSVs to DeepMD-kit training data.

**Config sections read:** `paths.cg_data_base_dir`,
`paths.deepmd_output_base_dir`, `deepmd`, `processing`, `output`, `simulations`.

**CLI:**
```
--base-dir DIR       override paths.cg_data_base_dir      (CG CSV input)
--output-dir DIR     override paths.deepmd_output_base_dir (DeepMD output)
--sim NAME [NAME...] filter simulations
--temp K [K...]      override temperatures
--workers N          parallel workers
```

**Outputs (per sim/temp):** `box.raw`, `coord.raw`, `force.raw`,
`atom_ener.raw`, `energy.raw`, `type.raw`, `type_map.raw`, and
`set.000/{box,coord,force,atom_ener,energy}.npy`.

---

### `cgkit fparam extract`  *(legacy `generate_fparam.py extract`)*

Parses LAMMPS `log.lammps` per-frame temperatures and writes DeepMD-kit
`fparam.raw` / `set.000/fparam.npy`. Temperatures are converted to eV
(`T * k_B`, `k_B = 8.617333262e-5 eV/K`) unless `--unit K` is requested.

**Config sections read:** `paths.log_dir`, `paths.deepmd_output_base_dir`,
`fparam.unit`, `fparam.extract.sim_names`.

**CLI:**
```
--sim NAME [NAME...] sims to process  (default from fparam.extract.sim_names)
--unit {K,eV}        output unit      (default from fparam.unit)
--log-dir DIR        override paths.log_dir
--output-dir DIR     override paths.deepmd_output_base_dir
```

---

### `cgkit fparam const`  *(legacy `generate_fparam.py const`)*

Writes constant-T `fparam.raw` / `.npy` matched frame-for-frame to existing
`box.raw` files (used for NPT/NVT equilibrated data where T is fixed).

**Config sections read:** `paths.deepmd_output_base_dir`, `fparam.unit`,
`fparam.const.{sim_names,temperatures}`.

**CLI:**
```
--sim NAME [NAME...] sims to process  (default from fparam.const.sim_names)
--temp K [K...]      temperatures     (default from fparam.const.temperatures)
--unit {K,eV}        output unit
--base-dir DIR       override paths.deepmd_output_base_dir
```

---

### `cgkit analyze-cg`  *(legacy `0x-analyze_cg_data.py`)*

Statistical analysis of CG CSV data: per-frame energy / particle-count
distributions, per-temperature breakdowns, time-series, and overview figures.

**Config sections read:** `paths.cg_data_base_dir`, `analysis_cg`.

**CLI:**
```
--base-dir DIR       override paths.cg_data_base_dir
--output-dir DIR     override analysis_cg.output_dir
--sim/--temp/--workers common
```

**Outputs (in `analysis_cg.output_dir`):** CSV summary tables + PNG figures.

---

### `cgkit analyze-atomic`  *(legacy `0x-analyze_atomic_structure.py`)*

SOAP descriptors → PCA → t-SNE → **UMAP** → clustering (DBSCAN/KMeans) pipeline,
with optional PyTorch GNN embeddings and graph topology visualization. Operates
on either CG trajectories (`--mode cg`, reads `*_cg.lammpstrj`) or atomic dumps
(`--mode aa`). UMAP is an optional step (requires `pip install -e ".[umap]"`);
when `umap-learn` is unavailable the pipeline prints a warning and skips UMAP
without affecting PCA/t-SNE/clustering.

**Config sections read:** `analysis_atomic.*` (`soap`, `pca`, `tsne`, `umap`,
`clustering`, `gnn_*`, `max_frames`, `max_per_file`, `output_dir`),
`paths.{cg,aa}_data_base_dir`, `simulations`. The new `analysis_atomic.umap`
subsection accepts `n_components` (default 5), `n_neighbors` (15), `min_dist`
(0.1), `metric` (`'euclidean'`); `analysis_atomic.clustering.space` selects
the projection fed to clustering (`pca` / `tsne` / `umap`, default `pca`).

**CLI:**
```
--mode {cg,aa}            analysis mode (default from config)
--base-dir DIR            override CG/AA base dir (chosen by --mode)
--output-dir DIR          override analysis_atomic.output_dir
--max-frames N            cap total frames
--max-per-file N          cap frames per trajectory file (CG mode)
--cluster-space {pca,tsne,umap}  projection fed to clustering
                                  (overrides analysis_atomic.clustering.space)
--sim/--temp/--workers    common
```

**Outputs:** `pca_results.csv`, `tsne_results.csv`, `umap_results.csv`,
`descriptors.csv`, `outlier_structures.csv`, plus PNG figures under `figures/`
(including `umap_overall.png` / `umap_<sim>.png` when UMAP runs). In CG mode
the files are prefixed `CG_` (e.g. `CG_umap_results.csv`). Since v2 all CSVs
end with the tracing columns `structure_id, source_file, temp` so any point in
PCA/t-SNE/UMAP/cluster space can be mapped back to its original dump frame
(`structure_id` format: `<sim>/<temp|ramp>@<timestep>`).

---

### `cgkit plot-pt`  *(new module)*

Joins every AA dump frame to its thermo row in `<sim>/log.lammps` and renders
a single P-vs-T scatter, colored by `sim_type`. Useful for spotting holes in
the (P, T) coverage of the training set before fitting a CG potential. Uses
`cglib.fparam.parse_lammps_thermo` + `query_thermo` to resolve `Temp`/`Press`
per timestep across multi-block logs (handles `reset_timestep` between
temperature sweeps).

**Config sections read:** `plot_pt.*` (`output_dir`, `max_frames`),
`paths.{aa_data_base_dir,log_dir}`, `analysis_atomic.max_frames` (fallback).

**CLI:**
```
--base-dir DIR       override paths.aa_data_base_dir (AA dump root)
--output-dir DIR     override plot_pt.output_dir (CSV + PNG destination)
--log-dir DIR        override paths.log_dir (root holding <sim>/log.lammps)
--max-frames N       cap total frames plotted (uniform downsample)
--sim/--temp         filter by sim_type / path-derived nominal T
```

**Outputs** (under `<output_dir>/pt_overview/`):
- `pt_data.csv` — one row per frame with columns
  `structure_id, sim_type, temp_nominal, temp_measured, pressure, timestep, source_file`
- `figures/pt_overview.png` — Nature-style scatter (Arial, 183 mm wide,
  red-blue categorical palette, frameless legend)

---

### `cgkit select-structures`  *(new module)*

Re-clusters a `pca_results.csv` / `tsne_results.csv` / `umap_results.csv`
(produced by `cgkit analyze-atomic`) and, for each cluster, picks **N
maximally-spread structures** via maximin (farthest-point) sampling. Each
selected frame is extracted from its source LAMMPS dump into a standalone
`.lammpstrj` in the output directory, plus a `selection_manifest.csv` records
every pick. This is the standard workflow for selecting a diverse,
representative training subset.

**Workflow:** `cgkit analyze-atomic` (once, produces the projection CSV) →
`cgkit select-structures` (iterate on `--n-clusters` / `--n` freely; no need
to re-run the expensive SOAP/PCA pipeline).

**Config sections read:** `select_structures.*`
(`input`, `output_dir`, `method`, `space`, `n_clusters`, `min_samples`,
`include_noise`, `seed`).

**CLI:**
```
--input FILE         input CSV (pca_results.csv / tsne_results.csv /
                     umap_results.csv / CG variants)
--output-dir DIR     where selected dumps + selection_manifest.csv go
--n N                (required) structures to pick per cluster
--space {pca,tsne,umap}   projection space (default: auto-detect from
                          columns; priority PCA > UMAP > t-SNE)
--method {kmeans,dbscan}   default kmeans
--n-clusters K       KMeans cluster count (default 8)
--eps FLOAT          DBSCAN epsilon (default: 30th pct of pairwise distances)
--min-samples N      DBSCAN min_samples (default 5)
--include-noise      treat DBSCAN noise (label -1) as a selectable cluster
--seed N             KMeans random seed (default 42)
```

**Outputs** (under `<output_dir>/`):
- One `.lammpstrj` per selected frame — filename = sanitized `structure_id`
  (e.g. `1-npt_200_100000.lammpstrj`), containing that single timestep's
  atoms/box in standard LAMMPS dump format.
- `selection_manifest.csv` — columns
  `structure_id, cluster, selection_rank, source_file, timestep, temp,
  <PC1..PCk or tSNE1..tSNEk>, output_file`.

**Algorithm:** within each cluster, the first pick is the point nearest the
cluster centroid (deterministic); each subsequent pick maximises the minimum
distance to already-chosen points (classic FPS / maximin). Clusters smaller
than N contribute all their members (with a `[small-cluster]` notice).

---

### `cgkit cg-verify`  *(new module)*

Cross-checks CG particle CSV files (`*_particles.csv`) against their source
LAMMPS atomic dumps. Designed to catch CG-generation bugs — especially the
classic "wrapped-coordinate pattern matching picks atoms on opposite sides
of a periodic boundary" failure mode.

Two modes:

- **`--mode auto` (default)** — runs four checks per CG CSV:
  1. **`pbc`** — flags CG particles whose member atoms span > `--pbc-thresh`
     (default 45%) of a periodic box length. Warns at > 22.5%. Skips
     `manual_assignment=True` rows (manual assignments may legitimately
     cross boundaries).
  2. **`conservation`** — re-computes each CG particle's position (center
     atom copy), force, and PE from the atomic data using the same formula
     as `cg_gen.create_cg_particle`, and compares against the stored CSV
     values. Tolerances: `--force-tol` (default 1e-4 eV/Å), `--pe-tol`
     (default 1e-6 eV), position compared via minimum-image PBC difference
     (so stored wrapped or unwrapped coordinates both validate correctly).
     When `coarse_graining.unwrap_pbc=true`, the atomic frame is unwrapped
     with the same chain logic before recompute.
  3. **`coverage`** — every atomic row must appear in exactly one CG
     particle's `atom_indices`. Reports missing atoms and duplicates.
  4. **`manual`** — every entry in `coarse_graining.cg_assignments` must
     correspond to exactly one `manual_assignment=True` CG row with
     matching atom IDs and CG type; undeclared manual rows are reported
     as warnings.
- **`--mode manual --atoms ID1 ID2 ...`** — for each user-supplied atomic
  ID, prints which CG particle owns it, the sibling member IDs, the CG
  position, the atom's own position, and a quick PBC-span flag. Useful for
  tracking down specific atoms flagged by `--mode auto`.

**File discovery:** by default verifies the first sorted `*_particles.csv`
under `paths.cg_data_base_dir/<sim>/` (after `--sim`/`--temp` filtering).
Pass `--all` to scan every file, `--file PATH` to override, or
`--max-files N` to cap an `--all` scan.

**Atomic source resolution:** strips `_particles.csv` from the CG filename
and searches under `paths.aa_data_base_dir` (override with `--atomic-dir`)
in the sim's `trajectory_dir` / `data_subdir` / `output_subdir`, then
recursively as a fallback.

**Config sections read:** `verify_cg.*`
(`output_dir`, `checks`, `force_tolerance`, `pe_tolerance`,
`pbc_span_threshold`), `paths.{cg,aa}_data_base_dir`,
`coarse_graining.{position_source,average_forces,average_potential_energy,cg_assignments,unwrap_pbc,r_cutoff,id_patterns}`,
`simulations`.

**CLI:**
```
--mode {auto,manual}        auto = full 4-check scan (default);
                            manual = look up --atoms IDs
--atoms ID [ID...]          atomic IDs to look up (manual mode only)
--file PATH | --all         single explicit file (mutex) | scan all enabled sims
--max-files N               cap an --all scan
--base-dir DIR              override paths.cg_data_base_dir (CG CSV root)
--atomic-dir DIR            override paths.aa_data_base_dir (atomic dump root)
--output-dir DIR            where to write cg_verify_report.csv
--checks C [C...]           subset of {pbc,conservation,coverage,manual}
--force-tol TOL             force recompute tolerance in eV/Å (default 1e-4)
--pe-tol TOL                PE recompute tolerance in eV (default 1e-6)
--pbc-thresh FRAC           PBC-span FAIL threshold as fraction of L (default 0.45)
--no-csv                    skip writing cg_verify_report.csv
--failures-only             only write FAIL rows to the CSV report
--quiet, -q                 only print per-file block when a file has FAILs
--sim/--temp/--workers      common
```

**Outputs:**
- stdout: per-file human-readable report (frame-by-frame FAIL/WARN counts,
  worst-case recompute errors) + final summary.
- `<output_dir>/cg_verify_report.csv` (only when there are issues): one
  row per issue with columns `file, sim, temp, timestep, check, severity,
  cg_id, message, n_atoms, member_atom_ids, force_err, pe_err, pos_err,
  pbc_span_frac_{x,y,z}`.

**Exit codes:**
- `0` — all checks pass (WARNs allowed)
- `1` — at least one FAIL
- `2` — file-level error (atomic source missing / unparseable / timestep
  mismatch) prevents any verification
- `3` — CLI usage error (e.g. `--mode manual` without `--atoms`)

---

## 4. Configuration (`config.json`)

The unified config has 8 top-level sections (plus `description`/`version`):

```jsonc
{
  "paths": {
    "base_dir":                 ".../01.aa",                  // cg-gen input
    "cg_data_base_dir":         ".../02.cg_dataset",          // cg-gen output / to-deepmd / analyze-cg input
    "deepmd_output_base_dir":   ".../03.cg_npy/training_data",// to-deepmd / fparam output
    "aa_data_base_dir":         ".../01.aa",                  // analyze-atomic aa-mode input
    "log_dir":                  ".../01.aa",                  // fparam extract LAMMPS logs
    "analysis_output_base_dir": ".../0x.cgdata_analysis"      // analysis output root
  },
  "simulations":          [ /* 1-npt, 2-nvt, 3-upT, 4-dnT */ ],
  "coarse_graining":      { /* method, patterns, center_atom_type, ... */ },
  "deepmd":               { /* num_groups, use_type_column */ },
  "fparam":               { /* unit, extract.sim_names, const.{sim_names,temperatures} */ },
  "analysis_cg":          { /* sample, max_files, output_dir */ },
  "analysis_atomic":      { /* mode, max_frames, max_per_file, soap, pca, tsne, umap, clustering{space,...}, gnn_* */ },
  "plot_pt":              { /* output_dir, max_frames */ },
  "select_structures":    { /* input, output_dir, method, space, n_clusters, min_samples, include_noise, seed */ },
  "verify_cg":            { /* output_dir, checks, force_tolerance, pe_tolerance, pbc_span_threshold */ },
  "processing":           { /* parallel, max_workers, trajectory_filter */ },
  "output":               { /* save_particles, save_box_vectors, save_raw_files, save_npy_files */ }
}
```

### CLI override mapping

`cgkit` writes CLI overrides into the *right* config key per subcommand:

| CLI flag             | cg-gen                   | to-deepmd                   | fparam extract              | fparam const                 | analyze-cg                 | analyze-atomic                | plot-pt                       | select-structures             |
|----------------------|--------------------------|-----------------------------|-----------------------------|------------------------------|----------------------------|-------------------------------|-------------------------------|-------------------------------|
| `--input FILE`       | _(n/a)_                  | _(n/a)_                     | _(n/a)_                     | _(n/a)_                      | _(n/a)_                    | _(n/a)_                       | _(n/a)_                       | `select_structures.input`     |
| `--base-dir DIR`     | `paths.base_dir`         | `paths.cg_data_base_dir`    | _(n/a)_                     | `paths.deepmd_output_base_dir` | `paths.cg_data_base_dir`  | `paths.{cg,aa}_data_base_dir` | `paths.aa_data_base_dir`      | _(n/a)_                       |
| `--output-dir DIR`   | `paths.cg_data_base_dir` | `paths.deepmd_output_base_dir` | `paths.deepmd_output_base_dir` | _(n/a)_                  | `analysis_cg.output_dir`   | `analysis_atomic.output_dir`  | `plot_pt.output_dir`          | `select_structures.output_dir`|
| `--log-dir DIR`      | _(n/a)_                  | _(n/a)_                     | `paths.log_dir`             | _(n/a)_                      | _(n/a)_                    | _(n/a)_                       | `paths.log_dir`               | _(n/a)_                       |

Mapping lives in `cglib/config.py::COMMAND_PATH_OVERRIDES`.

`cgkit cg-verify` resolves its own overrides inside `cg_verify._resolve_settings`
(same convention, cg-verify-specific flag set): `--base-dir` →
`paths.cg_data_base_dir`, `--atomic-dir` → `paths.aa_data_base_dir`,
`--output-dir` → `verify_cg.output_dir`.

`cgkit cg-gen` resolves three further overrides inside `cg_gen.run` (not via
`COMMAND_PATH_OVERRIDES` because they target `coarse_graining.*` rather than
`paths.*`): `--r-cutoff` → `coarse_graining.r_cutoff`, `--unwrap-pbc` /
`--no-unwrap-pbc` → `coarse_graining.unwrap_pbc`.

---

## 5. Library API (`cglib/`)

Every domain module exposes a `run(config: dict, args: argparse.Namespace) -> int`.
Top-level helpers used inside the package are also importable:

```python
from cglib.config       import load_config, merge_config_with_args, get_section
from cglib.cli          import build_parser, add_common_args
from cglib.paths        import substitute_temp, glob_with_temp, find_paired_csvs
from cglib.parallel     import run_parallel                       # tasks, worker, n_workers, parallel, desc
from cglib.io_utils     import read_particles_csv, write_particles_csv, write_raw, write_npy, ...
from cglib.lammps       import LammpsDumpReader, LammpsFrame
from cglib.cg_gen       import coarse_grain_trajectory, process_simulation
from cglib.deepmd_conv  import process_simulation as to_deepmd_process_simulation
from cglib.fparam       import run_extract, run_const, parse_lammps_log
from cglib.analyze_cg   import CGDataAnalyzer
from cglib.analyze_atomic import AtomicStructureAnalyzer, StructureDescriptor, GraphNeuralNetwork
from cglib.cg_verify    import (verify_pbc_span, verify_conservation,
                                verify_coverage, verify_manual_fidelity)
```

**LAMMPS reader API (unified):**

```python
from cglib.lammps import LammpsDumpReader

reader = LammpsDumpReader("dump.lammpstrj")
reader.parse_file()
df         = reader.get_dataframe(timestep_index=-1)   # pandas DataFrame (cg-gen shape)
first      = reader.read_first_frame()                  # dict of ndarrays (AA shape)
all_frames = reader.read_all_frames()                   # list[dict] (CG shape)
```

The three adapter methods reproduce the exact return shapes the three legacy
readers (`LAMMPSTrajectoryParser`, `LAMMPSTrajectoryReader`,
`CGTrajectoryReader`) produced, so downstream domain logic is unchanged.

**Parallel worker pattern:**

```python
from cglib.parallel import run_parallel

results = run_parallel(
    tasks, my_worker_fn,
    n_workers=4, parallel=True,
    desc="coarse-graining", unit="file",
)
# results: list of (task, ok: bool, error_msg: str, result: dict)
```

---

## 6. Workflow example (end-to-end PE CG)

```bash
cd cgkit

# Step 1: atomic LAMMPS dumps -> CG CSVs (one _particles.csv + _box_vectors.csv per dump)
python cgkit.py cg-gen --workers 8

# Step 2: CG CSVs -> DeepMD-kit .raw / set.000/*.npy
python cgkit.py to-deepmd --workers 8

# Step 3: fparam for the temperature-ramped datasets (3-upT, 4-dnT)
python cgkit.py fparam extract

# Step 3': fparam for the constant-T NPT/NVT datasets (200K-600K)
python cgkit.py fparam const

# Step 4: quick statistical overview of the CG dataset
python cgkit.py analyze-cg

# Step 5: SOAP / PCA / t-SNE on the CG lammpstrj dumps
python cgkit.py analyze-atomic --mode cg --max-frames 500
```

For an all-atom analysis instead:

```bash
python cgkit.py analyze-atomic --mode aa --max-frames 200
```

---

## 7. Legacy scripts

The five original scripts are preserved (read-only reference) under `legacy/`:

```
legacy/
├── 02-get_CGdata_parall.py          -> cgkit cg-gen      / cglib.cg_gen
├── 03-trans_CGnpy_parall.py         -> cgkit to-deepmd   / cglib.deepmd_conv
├── generate_fparam.py               -> cgkit fparam      / cglib.fparam
├── 0x-analyze_cg_data.py            -> cgkit analyze-cg  / cglib.analyze_cg
└── 0x-analyze_atomic_structure.py   -> cgkit analyze-atomic / cglib.analyze_atomic
```

They are kept so the original behavior can be diff-compared against the
unified toolkit. They are **not** maintained — bug fixes go into `cglib/`.

---

## 8. Migration table (old call -> new call)

| Old invocation                                                    | New invocation                                            |
|-------------------------------------------------------------------|-----------------------------------------------------------|
| `python 02-get_CGdata_parall.py`                                   | `python cgkit.py cg-gen`                                  |
| `python 02-get_CGdata_parall.py --sim 1-npt --temp 200 300`        | `python cgkit.py cg-gen --sim 1-npt --temp 200 300`       |
| `python 03-trans_CGnpy_parall.py`                                  | `python cgkit.py to-deepmd`                               |
| `python generate_fparam.py extract`                                | `python cgkit.py fparam extract`                          |
| `python generate_fparam.py const`                                  | `python cgkit.py fparam const`                            |
| `python generate_fparam.py const --unit K`                         | `python cgkit.py fparam const --unit K`                   |
| `python 0x-analyze_cg_data.py`                                     | `python cgkit.py analyze-cg`                              |
| `python 0x-analyze_atomic_structure.py`                            | `python cgkit.py analyze-atomic --mode aa`                |
| `python 0x-analyze_atomic_structure.py --cg-mode`                  | `python cgkit.py analyze-atomic --mode cg`                |
| `python 0x-analyze_atomic_structure.py --max-frames 200 --max-per-file 5 --cg-mode` | `python cgkit.py analyze-atomic --mode cg --max-frames 200 --max-per-file 5` |

---

## 9. Troubleshooting

**`ModuleNotFoundError: No module named 'sklearn'`** (or `matplotlib`,
`scipy`, `networkx`, `ase`, `dscribe`, `torch`)
You ran `analyze-cg` or `analyze-atomic` without the optional extras. See
§1 for the per-subcommand dependency matrix.

**`Config file not found: ...config.json`**
Pass `--config /path/to/config.json` explicitly, or run from a directory whose
`cgkit/config.json` exists.

**`[skip] Log not found` in `fparam extract`**
The `paths.log_dir` / `--log-dir` directory does not contain
`<sim_name>/log.lammps`. Check the path.

**`pickle.PicklingError` in parallel workers**
Domain `_worker` functions are top-level in their modules; if you add new
workers, do not nest them inside another function — `ProcessPoolExecutor`
requires picklable callables.

**`Empty data after concatenation` in `to-deepmd`**
The CG CSV files were not found under `paths.cg_data_base_dir/<sim>/<temp>/`.
Run `cgkit cg-gen` first, or fix `data_subdir` in `config.json`.

**Heavy deps imported when running `cgkit cg-gen`**
This should not happen. If it does, check that you have not added an
`import cglib.analyze_atomic` or `import cglib.analyze_cg` at module top in
any non-analysis module. The lazy-load discipline is enforced by tests in
the migration plan; both `analyze_*` modules defer heavy imports to
`_import_deps()` / `_import_heavy_deps()` inside `run()`.

---

## 10. Design notes (for maintainers)

- **Empty `cglib/__init__.py`** guarantees `import cglib` is side-effect-free.
- **Lazy trampoline for `analyze-atomic`** in `cgkit.py::_run_analyze_atomic`
  keeps the top-level dispatcher free of any transitive heavy import.
- **Unified LAMMPS parser** (`cglib/lammps.py`) replaces three legacy readers
  via adapter methods (`get_dataframe` / `read_first_frame` /
  `read_all_frames`) that preserve the original return shapes.
- **Domain logic 1:1**: coarse-graining patterns, SOAP fallbacks, DBSCAN eps
  percentile, t-SNE parameters, plot formats, CSV column orders — all
  preserved exactly from the legacy scripts. Behavior changes go through a
  `diff -r` parity check against `legacy/` outputs.
