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
| `analyze-cg`         | `matplotlib`, `scipy`                                                    |
| `analyze-atomic`     | `matplotlib`, `scipy`, `scikit-learn`, `networkx` (optional)             |
| `analyze-atomic` (SOAP)  | + `ase`, `dscribe`                                                   |
| `analyze-atomic` (GNN)   | + `torch`, `torch-geometric` (falls back to random embeddings)       |

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
```

**Per-trajectory outputs:** `<basename>_particles.csv`,
`<basename>_box_vectors.csv`, optionally `<basename>_cg.lammpstrj`.

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

SOAP descriptors → PCA → t-SNE → clustering (DBSCAN/KMeans) pipeline, with
optional PyTorch GNN embeddings and graph topology visualization. Operates on
either CG trajectories (`--mode cg`, reads `*_cg.lammpstrj`) or atomic dumps
(`--mode aa`).

**Config sections read:** `analysis_atomic.*` (`soap`, `pca`, `tsne`,
`clustering`, `gnn_*`, `max_frames`, `max_per_file`, `output_dir`),
`paths.{cg,aa}_data_base_dir`, `simulations`.

**CLI:**
```
--mode {cg,aa}       analysis mode (default from config)
--base-dir DIR       override CG/AA base dir (chosen by --mode)
--output-dir DIR     override analysis_atomic.output_dir
--max-frames N       cap total frames
--max-per-file N     cap frames per trajectory file (CG mode)
--sim/--temp/--workers common
```

**Outputs:** `pca_results.csv`, `tsne_results.csv`, `descriptors.csv`,
`outlier_structures.csv`, plus PNG figures under `figures/`. Since v2 all
CSVs end with the tracing columns `structure_id, source_file, temp` so any
point in PCA/t-SNE/cluster space can be mapped back to its original dump
frame (`structure_id` format: `<sim>/<temp|ramp>@<timestep>`).

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
  "analysis_atomic":      { /* mode, max_frames, max_per_file, soap, pca, tsne, clustering, gnn_* */ },
  "processing":           { /* parallel, max_workers, trajectory_filter */ },
  "output":               { /* save_particles, save_box_vectors, save_raw_files, save_npy_files */ }
}
```

### CLI override mapping

`cgkit` writes CLI overrides into the *right* config key per subcommand:

| CLI flag             | cg-gen                   | to-deepmd                   | fparam extract              | fparam const                 | analyze-cg                 | analyze-atomic                | plot-pt                       |
|----------------------|--------------------------|-----------------------------|-----------------------------|------------------------------|----------------------------|-------------------------------|-------------------------------|
| `--base-dir DIR`     | `paths.base_dir`         | `paths.cg_data_base_dir`    | _(n/a)_                     | `paths.deepmd_output_base_dir` | `paths.cg_data_base_dir`  | `paths.{cg,aa}_data_base_dir` | `paths.aa_data_base_dir`      |
| `--output-dir DIR`   | `paths.cg_data_base_dir` | `paths.deepmd_output_base_dir` | `paths.deepmd_output_base_dir` | _(n/a)_                  | `analysis_cg.output_dir`   | `analysis_atomic.output_dir`  | `plot_pt.output_dir`          |
| `--log-dir DIR`      | _(n/a)_                  | _(n/a)_                     | `paths.log_dir`             | _(n/a)_                      | _(n/a)_                    | _(n/a)_                       | `paths.log_dir`               |

Mapping lives in `cglib/config.py::COMMAND_PATH_OVERRIDES`.

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
