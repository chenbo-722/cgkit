# CH_CG 粗粒化工具包（cgkit）使用说明书

## 1. 概述

`cgkit` 是 **CH_CG PYTHON_tools** 套件的唯一入口，提供了一整套用于处理聚乙烯（PE）原子级 LAMMPS 轨迹的 Python 工具。其工作流程涵盖：

1. **粗粒化数据生成**：从原子级 LAMMPS 轨迹（`*.lammpstrj`）提取粗粒化粒子轨迹与盒子信息。
2. **DeepMD-kit 数据转换**：将粗粒化 CSV 数据转换为 DeepMD-kit 训练所需的 `.raw` / `.npy` 格式。
3. **温度参数生成**：提取或构造每帧温度参数（`fparam`），用于描述体系温度随时间的变化。
4. **统计与结构分析**：对粗粒化数据集进行能量分布、粒子数统计、时间序列分析。
5. **高维结构表征**：基于 SOAP 描述符 → PCA → t-SNE → 聚类（DBSCAN/KMeans）的完整降维与聚类流程，支持可选的 PyTorch GNN 图嵌入分析。

**设计优势**：
- 统一入口（`cgkit.py`）替代原先 5 个独立脚本（约 5,200 行），保留原始算法 1:1 还原。
- 延迟加载（Lazy-load）重型依赖：仅需 `numpy`、`pandas`、`tqdm` 即可运行核心流程；分析类命令仅在首次使用时加载 `matplotlib`、`scipy`、`torch` 等。
- 统一的 JSON 配置与 CLI 覆盖机制，所有路径与参数均可在命令行即时覆盖。

---

## 2. 安装与依赖

### 2.1 基础依赖（所有子命令必需）

```bash
pip install numpy pandas tqdm
```

### 2.2 可选依赖（按功能按需安装）

| 子命令 | 额外依赖 |
| :--- | :--- |
| `cg-gen` | 无 |
| `to-deepmd` | 无 |
| `fparam extract / const` | 无 |
| `analyze-cg` | `matplotlib`, `scipy` |
| `analyze-atomic` | `matplotlib`, `scipy`, `scikit-learn`；`networkx`（可选） |
| `analyze-atomic`（SOAP 描述符） | 额外需要 `ase`, `dscribe` |
| `analyze-atomic`（GNN 嵌入） | 额外需要 `torch`, `torch-geometric`（缺失时自动回退到随机嵌入） |

---

## 3. 快速开始：端到端 PE 粗粒化流程

```bash
cd PYTHON_tools

# 步骤 1：原子级 LAMMPS 轨迹 → 粗粒化 CSV 数据
python cgkit.py cg-gen --workers 8

# 步骤 2：粗粒化 CSV → DeepMD-kit 训练数据 (.raw / .npy)
python cgkit.py to-deepmd --workers 8

# 步骤 3：为变温模拟（升温/降温）提取逐帧温度参数
python cgkit.py fparam extract

# 步骤 3'：为恒温模拟（NPT/NVT）生成恒定温度参数
python cgkit.py fparam const

# 步骤 4：粗粒化数据的统计概览与可视化
python cgkit.py analyze-cg

# 步骤 5：基于 CG 轨迹的 SOAP/PCA/t-SNE/聚类结构分析
python cgkit.py analyze-atomic --mode cg --max-frames 500
```

---

## 4. 子命令详解

### 4.1 `cgkit cg-gen` — 粗粒化数据生成

- **功能**：读取 `paths.base_dir/<sim>/traj/*.lammpstrj`，执行灵活的粗粒化映射（如模式 `[1,2,2]` 表示 1 个 C 原子 + 2 个 H 原子），输出粗粒化粒子文件与盒子信息。
- **对应旧脚本**：`02-get_CGdata_parall.py`
- **配置节**：`paths.base_dir`, `paths.cg_data_base_dir`, `coarse_graining`, `processing`, `output`, `simulations`
- **常用 CLI 参数**：
  - `--base-dir DIR`：覆盖原子级输入路径
  - `--output-dir DIR`：覆盖 CG CSV 输出路径
  - `--sim NAME [NAME...]`：筛选模拟（如 `1-npt`, `2-nvt`）
  - `--temp K [K...]`：覆盖温度列表
  - `--workers N`：并行进程数
- **输出**：每个轨迹生成 `<basename>_particles.csv`, `<basename>_box_vectors.csv`，可选 `<basename>_cg.lammpstrj`

### 4.2 `cgkit to-deepmd` — DeepMD-kit 数据转换

- **功能**：将 CG CSV 文件转换为 DeepMD-kit 训练数据集。
- **对应旧脚本**：`03-trans_CGnpy_parall.py`
- **配置节**：`paths.cg_data_base_dir`, `paths.deepmd_output_base_dir`, `deepmd`, `processing`, `output`, `simulations`
- **常用 CLI 参数**：与 `cg-gen` 类似，输入/输出路径分别对应 CG 数据与 DeepMD 输出目录。
- **输出**（按 `sim/temp` 组织）：
  - `box.raw`, `coord.raw`, `force.raw`, `atom_ener.raw`, `energy.raw`
  - `type.raw`, `type_map.raw`
  - `set.000/{box,coord,force,atom_ener,energy}.npy`

### 4.3 `cgkit fparam extract` — 逐帧温度提取

- **功能**：解析 `log.lammps`，提取每帧温度，写入 `fparam.raw` / `set.000/fparam.npy`。默认将温度转换为能量单位 eV（`T * k_B`，`k_B = 8.617333262e-5 eV/K`）。
- **对应旧脚本**：`generate_fparam.py extract`
- **配置节**：`paths.log_dir`, `paths.deepmd_output_base_dir`, `fparam.unit`, `fparam.extract.sim_names`
- **常用 CLI 参数**：
  - `--sim NAME ...`：指定模拟名称
  - `--unit {K,eV}`：输出单位（默认 eV）
  - `--log-dir DIR`：覆盖 LAMMPS 日志路径
  - `--output-dir DIR`：覆盖输出路径

### 4.4 `cgkit fparam const` — 恒定温度参数

- **功能**：为恒温（NPT/NVT）数据集生成与现有 `box.raw` 帧一一对应的恒定温度 `fparam`。
- **对应旧脚本**：`generate_fparam.py const`
- **配置节**：`paths.deepmd_output_base_dir`, `fparam.unit`, `fparam.const.{sim_names,temperatures}`
- **常用 CLI 参数**：
  - `--sim NAME ...`
  - `--temp K ...`：指定各模拟的恒定温度
  - `--unit {K,eV}`

### 4.5 `cgkit analyze-cg` — 粗粒化统计分析

- **功能**：对 CG CSV 数据进行统计：每帧能量、粒子数分布、按温度分组汇总、时间序列、概览图。
- **对应旧脚本**：`0x-analyze_cg_data.py`
- **配置节**：`paths.cg_data_base_dir`, `analysis_cg`
- **输出**：在 `analysis_cg.output_dir` 中生成 CSV 汇总表与 PNG 统计图。

### 4.6 `cgkit analyze-atomic` — 原子/粗粒化结构分析

- **功能**：SOAP 描述符 → PCA → t-SNE → 聚类（DBSCAN/KMeans）全流程；支持可选的 PyTorch GNN 图嵌入与网络拓扑可视化。
- **对应旧脚本**：`0x-analyze_atomic_structure.py`
- **配置节**：`analysis_atomic.*`, `paths.{cg,aa}_data_base_dir`, `simulations`
- **模式**：
  - `--mode cg`：分析粗粒化轨迹（读取 `*_cg.lammpstrj`）
  - `--mode aa`：分析全原子轨迹
- **常用 CLI 参数**：
  - `--max-frames N`：限制总帧数
  - `--max-per-file N`：限制每个轨迹文件的帧数（CG 模式）
- **输出**：
  - `pca_results.csv`, `tsne_results.csv`, `descriptors.csv`
  - `outlier_structures.csv`
  - `figures/` 目录下的 PNG 可视化图

---

## 5. 配置文件（`config.json`）

`config.json` 是统一配置中心，包含 8 个顶层配置节：

| 配置节 | 说明 |
| :--- | :--- |
| `paths` | 输入/输出根目录路径（原子数据、CG 数据、DeepMD 数据、日志、分析输出） |
| `simulations` | 模拟列表（如 `1-npt`, `2-nvt`, `3-upT`, `4-dnT`） |
| `coarse_graining` | 粗粒化方法、映射模式、中心原子类型等 |
| `deepmd` | DeepMD 数据格式参数（组数、类型列使用等） |
| `fparam` | 温度单位、extract/const 的模拟名称与温度列表 |
| `analysis_cg` | 采样数、最大文件数、输出目录等 |
| `analysis_atomic` | 分析模式、最大帧数、SOAP/PCA/t-SNE/聚类/GNN 参数 |
| `processing` | 并行开关、最大进程数、轨迹过滤规则 |
| `output` | 各阶段保存开关（粒子/盒子/RAW/NPY 文件） |

**CLI 覆盖规则**：命令行参数（如 `--base-dir`, `--output-dir`）会自动映射到对应子命令的正确配置键，无需手动修改 JSON。

---

## 6. 库 API（二次开发）

除命令行外，`cglib/` 包可直接导入用于脚本化开发：

```python
from cglib.config       import load_config, merge_config_with_args
from cglib.lammps       import LammpsDumpReader
from cglib.cg_gen       import coarse_grain_trajectory
from cglib.deepmd_conv  import process_simulation
from cglib.analyze_atomic import AtomicStructureAnalyzer
```

**统一 LAMMPS 读取器**：
```python
from cglib.lammps import LammpsDumpReader

reader = LammpsDumpReader("dump.lammpstrj")
reader.parse_file()

df         = reader.get_dataframe(timestep_index=-1)  # pandas DataFrame（CG 模式）
first      = reader.read_first_frame()                 # dict of ndarrays（AA 模式）
all_frames = reader.read_all_frames()                  # list[dict]（CG 模式）
```

---

## 7. 旧脚本迁移对照表

| 旧调用方式 | 新调用方式 |
| :--- | :--- |
| `python 02-get_CGdata_parall.py` | `python cgkit.py cg-gen` |
| `python 02-get_CGdata_parall.py --sim 1-npt --temp 200 300` | `python cgkit.py cg-gen --sim 1-npt --temp 200 300` |
| `python 03-trans_CGnpy_parall.py` | `python cgkit.py to-deepmd` |
| `python generate_fparam.py extract` | `python cgkit.py fparam extract` |
| `python generate_fparam.py const` | `python cgkit.py fparam const` |
| `python 0x-analyze_cg_data.py` | `python cgkit.py analyze-cg` |
| `python 0x-analyze_atomic_structure.py` | `python cgkit.py analyze-atomic --mode aa` |
| `python 0x-analyze_atomic_structure.py --cg-mode` | `python cgkit.py analyze-atomic --mode cg` |

---

## 8. 常见问题与排查

| 问题 | 原因 | 解决方案 |
| :--- | :--- | :--- |
| `ModuleNotFoundError: sklearn` / `matplotlib` / `scipy` 等 | 运行了分析命令但未安装可选依赖 | 根据第 2.2 节安装对应额外包 |
| `Config file not found: ...config.json` | 当前目录缺少 `config.json` | 显式传入 `--config /path/to/config.json`，或确保在 `PYTHON_tools` 目录下运行 |
| `[skip] Log not found`（`fparam extract`） | `paths.log_dir` 下不存在 `<sim>/log.lammps` | 检查路径配置与文件是否存在 |
| `pickle.PicklingError`（并行时） | 自定义的 `_worker` 函数嵌套在其它函数内部 | 确保并行 worker 函数定义在模块顶层，可被 `ProcessPoolExecutor` 序列化 |
| `Empty data after concatenation`（`to-deepmd`） | 未找到 CG CSV 文件 | 先运行 `cgkit cg-gen`，或修正 `config.json` 中的 `data_subdir` 路径 |
| 运行 `cg-gen` 时加载了重型包 | 非分析模块顶层误引入了 `cglib.analyze_*` | 检查代码导入，确保分析模块仅在 `run()` 内部延迟加载 |

---

## 9. 备注

- `legacy/` 目录中保留了 5 个原始脚本作为只读参考，用于行为对比与版本追溯，但不再维护；所有修复与优化均在 `cglib/` 与 `cgkit.py` 中进行。
- 重型依赖（`torch`, `scikit-learn`, `dscribe` 等）采用延迟导入策略，因此 `import cglib` 不会产生副作用，也不会强制安装分析包。
