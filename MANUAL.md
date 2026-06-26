# CH_CG 粗粒化工具包（cgkit）使用说明书

## 1. 概述

`cgkit` 是 **CH_CG cgkit** 套件的唯一入口，提供了一整套用于处理聚乙烯（PE）原子级 LAMMPS 轨迹的 Python 工具。其工作流程涵盖：

1. **粗粒化数据生成**：从原子级 LAMMPS 轨迹（`*.lammpstrj`）提取粗粒化粒子轨迹与盒子信息。
2. **DeepMD-kit 数据转换**：将粗粒化 CSV 数据转换为 DeepMD-kit 训练所需的 `.raw` / `.npy` 格式。
3. **温度参数生成**：提取或构造每帧温度参数（`fparam`），用于描述体系温度随时间的变化。
4. **统计与结构分析**：对粗粒化数据集进行能量分布、粒子数统计、时间序列分析。
5. **高维结构表征**：基于 SOAP 描述符 → PCA → t-SNE → 聚类（DBSCAN/KMeans）的完整降维与聚类流程，支持可选的 PyTorch GNN 图嵌入分析。
6. **P–T 空间覆盖**：将每个原子级 dump 帧的 timestep 关联至 `log.lammps` 的 thermo 行，输出 P–T 散点图与 `pt_data.csv`，便于评估训练集的温度/压力覆盖度。

**设计优势**：
- 统一入口（`cgkit.py`）替代原先 5 个独立脚本（约 5,200 行），保留原始算法 1:1 还原。
- 延迟加载（Lazy-load）重型依赖：仅需 `numpy`、`pandas`、`tqdm` 即可运行核心流程；分析类命令仅在首次使用时加载 `matplotlib`、`scipy`、`torch` 等。
- 统一的 JSON 配置与 CLI 覆盖机制，所有路径与参数均可在命令行即时覆盖。

---

## 2. 安装与依赖

### 2.1 安装 `cgkit` 命令（推荐）

在 `cgkit/` 项目目录下执行：

```bash
pip install -e .
```

这会在 conda/venv 的 `bin/` 中生成一个 `cgkit` 可执行文件，加入 `PATH`
后即可从**任意目录**调用：

```bash
cgkit cg-gen
cgkit plot-pt --max-frames 200
```

`config.json` 通过源码树相对路径自动定位（`cglib/config.py` 中的
`__file__` 解析），与当前工作目录无关。该安装为**可编辑模式**
（editable）——修改 `cgkit.py` / `cglib/` 后立即生效，无需重装。

按需安装可选依赖组（对应第 2.3 节的矩阵）：

```bash
pip install -e ".[atomic,soap]"     # analyze-atomic + SOAP 描述符
pip install -e ".[all]"             # 全部（重型：会拉取 torch、dscribe 等）
```

> 不想安装？也可直接 `cd cgkit && python cgkit.py <子命令>`，但必须在
> 项目目录内运行（`cglib` 需可被导入）。

### 2.2 基础依赖（所有子命令必需）

```bash
pip install numpy pandas tqdm
```

### 2.3 可选依赖（按功能按需安装）

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

> 以下假设已执行 `pip install -e .`（见 2.1）。未安装时，将 `cgkit` 替换
> 为 `python cgkit.py` 并确保在项目目录内运行。

```bash
# 步骤 1：原子级 LAMMPS 轨迹 → 粗粒化 CSV 数据
cgkit cg-gen --workers 8

# 步骤 2：粗粒化 CSV → DeepMD-kit 训练数据 (.raw / .npy)
cgkit to-deepmd --workers 8

# 步骤 3：为变温模拟（升温/降温）提取逐帧温度参数
cgkit fparam extract

# 步骤 3'：为恒温模拟（NPT/NVT）生成恒定温度参数
cgkit fparam const

# 步骤 4：粗粒化数据的统计概览与可视化
cgkit analyze-cg

# 步骤 5：基于 CG 轨迹的 SOAP/PCA/t-SNE/聚类结构分析
cgkit analyze-atomic --mode cg --max-frames 500
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

#### 4.1.1 粗粒化映射原理与规则

本节详述 `cg-gen` 背后的粗粒化（coarse-graining, CG）算法，对应实现位于
`cglib/cg_gen.py::coarse_grain_trajectory()`。算法从原子级 LAMMPS 轨迹中
按化学规则把若干原子归并为一个 CG 粒子（bead），并派生其位置、受力、势能
等属性，用于训练 CG 势能面。

##### A. 化学动机：聚乙烯的 CH₂/CH₃ 映射

本工作流的粗粒化目标是聚乙烯（PE，–(CH₂)ₙ–）链。全原子模型中：

- **type 1 = 碳（C）**，质量 12，构成主链；
- **type 2 = 氢（H）**，质量 1，键合在碳上。

自然的 CG 映射是把每个 CHₙ 基团塌缩为**一个 CG bead**，bead 的位置取自碳原子：

| 化学基团 | 原子组成 | 链上位置 | CG bead |
| :--- | :--- | :--- | :--- |
| CH₂ | 1 C + 2 H | 主链重复单元 | type 1 |
| CH₃ | 1 C + 3 H | 链端 | type 2 |

生产配置 `config.json` 正是按此映射设定：
- `patterns: [[1, 2, 2]]` 自动匹配主链 CH₂（占绝大多数）；
- `cg_assignments` 显式列出两个 CH₃ 链端（每个含 1 C + 3 H，共 4 个原子）。

##### B. 两阶段算法

`coarse_grain_trajectory()` 对每一帧按以下顺序处理：

**Step 1 — 手动指派（`cg_assignments`，优先级最高）**

逐条遍历配置中的 `cg_assignments`：

1. 按 `atom_ids` 在当前帧 DataFrame 中定位目标原子；
2. 若其中任一原子已被先前指派消费（`used_atom_indices`），则跳过该条；
3. 否则调用 `create_cg_particle()` 生成一个 CG bead，`type` 取自配置的
   `cg_type`，位置取该组**第一个原子**（即链端碳）的坐标；
4. 把这些原子标记为已消费。

Step 1 始终先于 Step 2 执行，因此可用来覆盖模式匹配（例如把链端 CH₃
强行指定为独立 type 2，而非被 [1,2,2] 误配成 CH₂）。

**Step 2 — 模式匹配（pattern-based）**

对剩下未被消费的所有中心原子（`type == center_atom_type`，默认 1 = C），
逐个尝试每个 pattern：

1. **统计邻居需求**：`pattern.count(2)` 给出该 pattern 需要多少个 type 2
   （H）邻居。例如 `[1, 2, 2]` 需要 2 个 H。
2. **找最近邻**：在未被消费的 type 2 原子中，用 `calculate_distance()`
   逐一计算到中心原子的（平方）距离，升序排序，取前 N 个作为候选。
3. **打分选最优 pattern**（见下节 4.1.1‑C）。
4. **生成 bead**：调用 `create_cg_particle()`，把这些原子标记为已消费。

中心原子的处理顺序由 pandas `iterrows()` 决定（按 DataFrame 行序）；
当 type 2 原子被某 bead 消费后，后续 center 原子无法再选中它——因此
**邻居是排他的**，不会出现两个 CG bead 共享同一个 H。

##### C. Pattern 选择规则与评分

当配置了多个 pattern（如同时给出 `[1,2,2]` 和 `[1,2,2,2]`）时，每个
中心原子按下式打分并取最高分 pattern：

```
score = len(selected_atoms) - pattern_length * 0.01
```

其中 `len(selected_atoms)` = 1（中心 C）+ N（选中的 H），恰好等于
`pattern_length`。因此等价于：

```
score = pattern_length * 0.99
```

**含义**：在所有能被满足（可用 H 数 ≥ N）的 pattern 中，**原子数更多
的 pattern 得分更高**，会被优先采纳。例如同时配置两种 pattern 时：

- 若碳周围有 ≥ 3 个可用 H → `[1,2,2,2]` 胜出（4 原子 bead）；
- 若可用 H 不足 3 个 → 回退到 `[1,2,2]`（3 原子 bead）。

> ⚠ **生产配置只声明 `[1, 2, 2]` 单一 pattern**。若同时声明两种 pattern，
> 评分机制会让所有能找到 3 个 H 的碳都变成 type 2（CH₃），这与 PE 主链
> 全为 CH₂ 的化学事实相悖。因此 CH₃ 链端通过 `cg_assignments` 显式
> 手动指派，主链 CH₂ 由单一 pattern 自动匹配——这是当前评分函数下
> 既正确又简洁的写法。

##### D. Pattern → CG type 的自动映射

CG bead 的 `type` 字段由 pattern 按**首次出现顺序**自增分配（见
`coarse_grain_trajectory()` 与 `run()` 的 Pattern→Type 映射表）：

```python
pattern_to_cg_type = {}
cg_type_counter = 1
for pattern in patterns:                      # 按 config 中的顺序
    if tuple(pattern) not in pattern_to_cg_type:
        pattern_to_cg_type[tuple(pattern)] = cg_type_counter
        cg_type_counter += 1
```

- `[1, 2, 2]` → Type 1（首个声明的 pattern）
- `[1, 2, 2, 2]` → Type 2（若存在）
- 手动指派的 bead 直接用配置中的 `cg_type`，**不参与**此自增映射。

运行 `cgkit cg-gen` 时终端会打印这张映射表，便于核对。

##### E. CG bead 的属性派生规则

`create_cg_particle()` 按如下规则生成每个 bead 的字段：

| 字段 | 来源 |
| :--- | :--- |
| `id` | 自增整数（1, 2, 3, …，帧内连续） |
| `type` | 来自 pattern→type 映射（Step 2）或 `cg_assignments.cg_type`（Step 1） |
| `x, y, z` | **中心原子**（C）的坐标；unwrapped 优先，否则 wrapped |
| `fx, fy, fz` | 见下方力/能规则 |
| `c_pe` | 见下方力/能规则 |
| `n_atoms` | 组内原子数（如 CH₂=3, CH₃=4） |
| `atom_indices` | 组内原子在 DataFrame 中的行号 |
| `pattern` | 匹配的 pattern（仅 Step 2 bead） |
| `manual_assignment` | `True` + `assigned_atom_ids`（仅 Step 1 bead） |

**力 / 势能规则**由 `force_energy_source` 控制（默认 `"average"`）：

- `"average"`（默认）：
  - 若 `average_forces=true`，`fx/fy/fz` = 组内所有原子分量的**算术平均**；
  - 若 `average_potential_energy=true`，`c_pe` = 组内所有原子 `c_pe` 的平均。
- `"center_only"`：直接取中心原子的 `fx/fy/fz/c_pe`，不做平均。

物理意义：CG bead 的受力应等于其代表的所有原子的**总受力**。这里存的是
**平均**而非求和——这是约定选择，DeepMD-kit 训练时需对应 CG 力标签的
定义保持一致（若需总受力，可在 CG 配置中关闭 average 或在后处理 ×N）。

##### F. 距离计算与周期性边界

`calculate_distance()` 返回**平方距离**（非开方，保留 legacy 行为以便
排序时免去 sqrt 开销），并使用**最小镜像约定（minimum-image PBC）**处理
周期性边界：

```
dx = dx - round(dx / lx) * lx     # 同理 dy, dz
return dx*dx + dy*dy + dz*dz
```

- `lx/ly/lz` 取自当前帧 `box_bounds` 的 `xhi - xlo` 等；
- 若 `position_source="unwrapped"` 但 dump 中无 `xu/yu/zu` 列，自动回退
  到 wrapped（`x/y/z`）。

**推荐使用 unwrapped 坐标**：wrapped 坐标在原子穿越周期边界时会发生
跳变，导致伪近距离，把本不相邻的 H 误选进同一个 bead。

##### G. 盒矢量的导出

每个时间步输出一行 9 元素对角盒矢量（非对角项 = 0，正交盒）：

```
[xlo_edge=0, Lx, 0,  0, ylo_edge=0, Ly, 0,  0, zlo_edge=0, Lz]
```

其中 `Lx = xhi - xlo`（原始盒边界之差），并在 `box_vectors.csv` 末尾
附 `timestep` 列。CG 轨迹 dump（`*_cg.lammpstrj`）的 `BOX BOUNDS` 段
则写成 `0 Lx` 形式（即把盒子原点移到坐标原点）。

##### H. 输出文件 schema

每个原子级轨迹 `<basename>.lammpstrj` 经 `cg-gen` 处理后产出：

**`<basename>_particles.csv`**（每个 CG bead 一行，跨所有帧拼接）：

| 列 | 含义 |
| :--- | :--- |
| `id, type, x, y, z` | bead 几何与类型 |
| `fx, fy, fz, c_pe` | 平均力与平均势能 |
| `n_atoms` | 组内原子数（CH₂=3, CH₃=4） |
| `atom_indices` | 组内原子行号（list） |
| `pattern` | 匹配的 pattern（仅 Step 2） |
| `manual_assignment, assigned_atom_ids` | 仅 Step 1 |
| `timestep` | 所属 LAMMPS 时间步 |

**`<basename>_box_vectors.csv`**（每帧一行）：

| 列 | 含义 |
| :--- | :--- |
| `xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz` | 9 元素盒矢量（正交盒非对角=0） |
| `timestep` | LAMMPS 时间步 |

**`<basename>_cg.lammpstrj`**（可选，`export_cg_trajectory=true` 时生成）：
标准 LAMMPS dump 格式，列 `id type xu yu zu fx fy fz`（或 wrapped 的
`x y z`），可用 OVITO / VMD 直接可视化验证 CG 映射是否正确。

##### I. 常用配置速查

**典型 PE 配置（与 `config.json` 一致）**：

```json
"coarse_graining": {
  "method": "flexible_pattern",
  "patterns": [[1, 2, 2]],
  "center_atom_type": 1,
  "position_source": "unwrapped",
  "force_energy_source": "average",
  "average_forces": true,
  "average_potential_energy": true,
  "export_cg_trajectory": true,
  "cg_trajectory_filename": "{basename}_cg.lammpstrj",
  "cg_assignments": [
    {"atom_ids": [1,   401,  402,  403], "cg_type": 2, "description": "CH3 chain end"},
    {"atom_ids": [400, 1200, 1201, 1202], "cg_type": 2, "description": "CH3 chain end"}
  ]
}
```

**改写指南**：

| 想要 | 修改 |
| :--- | :--- |
| 主链 bead 改为 CH₃ 风格（1C+3H） | `patterns: [[1,2,2,2]]`，并删除/调整 `cg_assignments` |
| bead 受力用碳的力而非平均 | `force_energy_source: "center_only"` |
| bead 受力 = 组内总力（非平均） | 关闭 `average_forces`，或后处理 ×`n_atoms` |
| 用 wrapped 坐标 | `position_source: "wrapped"`（不推荐，易选错邻居） |
| 多种 bead 类型共存 | 在 `patterns` 里列多条，注意评分偏好更长 pattern；必要时用 `cg_assignments` 锁定特殊原子 |

##### J. 算法局限与注意事项

1. **邻居排他性**：一个 H 只能归一个 bead。若配置的 pattern 总需求 H 数
   超过体系实际 H 数，部分碳会找不到足够邻居而被静默跳过——检查
   `_particles.csv` 的 bead 总数与预期是否一致。
2. **处理顺序敏感**：center 原子按 DataFrame 行序处理，先到先得。若需
   确定性优先级（如要求链端碳先匹配），用 `cg_assignments` 显式指定。
3. **正交盒假设**：盒矢量的导出与 CG dump 的 `BOX BOUNDS` 只支持正交盒
   （非对角 tilt 因子 `xy/xz/yz` 恒为 0）。若 MD 用了三斜盒（triclinic），
   需在 `lammps.py` 与 `cg_gen.py` 中扩展 tilt 支持。
4. **评分函数偏向长 pattern**：见 4.1.1‑C 的告警，多 pattern 时需谨慎。
5. **力的语义**：CSV 存的是平均力；DeepMD 训练标签需与此一致。

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
  - 自 v2 起，上述 CSV 末尾统一附带溯源列 `structure_id, source_file, temp`，其中 `structure_id` 形如 `<sim>/<temp|ramp>@<timestep>`，便于从 PCA/t-SNE/聚类空间的点回到原始 dump 帧。

### 4.7 `cgkit plot-pt` — P–T 空间覆盖图

- **功能**：将每个 AA dump 帧的 `timestep` 关联至其所属 `<sim>/log.lammps` 的 thermo 行，得到该帧的实测温度（`Temp`）与压力（`Press`），并渲染一张 Nature 风格的 P–T 散点图。可识别多 run 块、`reset_timestep` 重复的复杂 log（与 `cgkit fparam extract` 共用 `parse_lammps_thermo` + `query_thermo`）。
- **对应模块**：`cglib/pt_plot.py`（新增）
- **配置节**：`plot_pt.{output_dir, max_frames}`，`paths.{aa_data_base_dir, log_dir}`
- **常用 CLI 参数**：
  - `--base-dir DIR`：覆盖 AA dump 根目录
  - `--output-dir DIR`：覆盖输出目录
  - `--log-dir DIR`：覆盖 `log.lammps` 所在根目录
  - `--max-frames N`：限制总帧数（均匀降采样）
  - `--sim NAME …` / `--temp K …`：按 sim_type / 名义温度筛选
- **输出**（位于 `<output_dir>/pt_overview/`）：
  - `pt_data.csv`：每帧一行，列为 `structure_id, sim_type, temp_nominal, temp_measured, pressure, timestep, source_file`
  - `figures/pt_overview.png`：单张 P-vs-T 散点（Arial、183 mm 双栏宽、红蓝 categorical 配色、无 spine/网格）
- **典型用途**：在拟合 CG 势能前快速检查 (P, T) 空间的覆盖空白；`temp_measured` 与 `pressure` 为 `NaN` 表示该帧 timestep 未在 log 中命中，会在终端打印 `[skip-log]` / `[no-match]` 计数。

---

## 5. 配置文件（`config.json`）

`config.json` 是统一配置中心，包含 9 个顶层配置节：

| 配置节 | 说明 |
| :--- | :--- |
| `paths` | 输入/输出根目录路径（原子数据、CG 数据、DeepMD 数据、日志、分析输出） |
| `simulations` | 模拟列表（如 `1-npt`, `2-nvt`, `3-upT`, `4-dnT`） |
| `coarse_graining` | 粗粒化方法、映射模式、中心原子类型等 |
| `deepmd` | DeepMD 数据格式参数（组数、类型列使用等） |
| `fparam` | 温度单位、extract/const 的模拟名称与温度列表 |
| `analysis_cg` | 采样数、最大文件数、输出目录等 |
| `analysis_atomic` | 分析模式、最大帧数、SOAP/PCA/t-SNE/聚类/GNN 参数 |
| `plot_pt` | P–T 覆盖图的输出目录与最大帧数（`output_dir`, `max_frames`） |
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
| `Config file not found: ...config.json` | 当前目录缺少 `config.json` | 显式传入 `--config /path/to/config.json`，或确保在 `cgkit` 目录下运行 |
| `[skip] Log not found`（`fparam extract`） | `paths.log_dir` 下不存在 `<sim>/log.lammps` | 检查路径配置与文件是否存在 |
| `pickle.PicklingError`（并行时） | 自定义的 `_worker` 函数嵌套在其它函数内部 | 确保并行 worker 函数定义在模块顶层，可被 `ProcessPoolExecutor` 序列化 |
| `Empty data after concatenation`（`to-deepmd`） | 未找到 CG CSV 文件 | 先运行 `cgkit cg-gen`，或修正 `config.json` 中的 `data_subdir` 路径 |
| 运行 `cg-gen` 时加载了重型包 | 非分析模块顶层误引入了 `cglib.analyze_*` | 检查代码导入，确保分析模块仅在 `run()` 内部延迟加载 |

---

## 9. 备注

- `legacy/` 目录中保留了 5 个原始脚本作为只读参考，用于行为对比与版本追溯，但不再维护；所有修复与优化均在 `cglib/` 与 `cgkit.py` 中进行。
- 重型依赖（`torch`, `scikit-learn`, `dscribe` 等）采用延迟导入策略，因此 `import cglib` 不会产生副作用，也不会强制安装分析包。
