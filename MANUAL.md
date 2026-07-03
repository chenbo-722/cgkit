# CH_CG 粗粒化工具包（cgkit）使用说明书

## 1. 概述

`cgkit` 是 **CH_CG cgkit** 套件的唯一入口，提供了一整套用于处理聚乙烯（PE）原子级 LAMMPS 轨迹的 Python 工具。其工作流程涵盖：

1. **粗粒化数据生成**：从原子级 LAMMPS 轨迹（`*.lammpstrj`）提取粗粒化粒子轨迹与盒子信息。
2. **DeepMD-kit 数据转换**：将粗粒化 CSV 数据转换为 DeepMD-kit 训练所需的 `.raw` / `.npy` 格式。
3. **温度参数生成**：提取或构造每帧温度参数（`fparam`），用于描述体系温度随时间的变化。
4. **统计与结构分析**：对粗粒化数据集进行能量分布、粒子数统计、时间序列分析。
5. **高维结构表征**：基于 SOAP 描述符 → PCA → t-SNE → 聚类（DBSCAN/KMeans）的完整降维与聚类流程，支持可选的 PyTorch GNN 图嵌入分析。
6. **P–T 空间覆盖**：将每个原子级 dump 帧的 timestep 关联至 `log.lammps` 的 thermo 行，输出 P–T 散点图与 `pt_data.csv`，便于评估训练集的温度/压力覆盖度。
7. **结构筛选**：消费 `analyze-atomic` 的 PCA/t-SNE 结果，在其投影空间上聚类，对每个类用 maximin（最远点采样）选 N 个最具多样性的构型，并把选中帧从源 dump 抽取成独立 LAMMPS dump 文件复制到指定目录，用于构造代表性训练子集。
8. **模型验证**：对比 DeepMD 模型预测值与参考数据，生成 parity 对比图（力/能量）与误差分布直方图，输出 RMSE/MAE/R² 汇总指标。

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
| `cg-verify` | 无 |
| `plot-test` | `matplotlib` |
| `analyze-cg` | `matplotlib`, `scipy` |
| `analyze-atomic` | `matplotlib`, `scipy`, `scikit-learn`；`networkx`（可选） |
| `analyze-atomic`（SOAP 描述符） | 额外需要 `ase`, `dscribe` |
| `analyze-atomic`（GNN 嵌入） | 额外需要 `torch`, `torch-geometric`（缺失时自动回退到随机嵌入） |
| `analyze-atomic`（UMAP 降维） | 额外需要 `umap-learn>=0.5`（缺失时自动跳过 UMAP 步骤，不影响 PCA/t-SNE/聚类） |

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
  - `--r-cutoff Å`：pattern 距离匹配截断半径（默认 1.25；`null` 表示不限）
  - `--unwrap-pbc` / `--no-unwrap-pbc`：开/关 cg-gen 前的链式 PBC unwrap 预处理
- **输出**：每个轨迹生成 `<basename>_particles.csv`, `<basename>_box_vectors.csv`，可选 `<basename>_cg.lammpstrj`。`_particles.csv` 自升级后多出 `match_status`（`manual` / `id_pattern` / `pattern`）与 `id_pattern`（命中的规则序列化）两列，便于溯源每个 bead 的来源阶段。

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
  - 若 `average_forces=true`，`fx/fy/fz` = 组内所有原子分量的**质量加权平均**
    （type 1 / C 质量 12，type 2 / H 质量 1）：∑(m_i · F_i) / ∑m_i；
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

- **功能**：SOAP 描述符 → PCA → t-SNE → **UMAP** → 聚类（DBSCAN/KMeans）全流程；支持可选的 PyTorch GNN 图嵌入与网络拓扑可视化。UMAP 为可选步骤（需 `pip install -e ".[umap]"`），`umap-learn` 缺失时自动跳过 UMAP，不影响其余步骤。
- **对应旧脚本**：`0x-analyze_atomic_structure.py`
- **配置节**：`analysis_atomic.*`, `paths.{cg,aa}_data_base_dir`, `simulations`
  - `analysis_atomic.umap`：UMAP 超参，键 `n_components`（默认 5，面向聚类；可视化只画前 2 维）、`n_neighbors`（15）、`min_dist`（0.1）、`metric`（`'euclidean'`）。
  - `analysis_atomic.clustering.space`：决定聚类读取的投影空间，取值 `pca` / `tsne` / `umap`，默认 `pca`（与升级前一致）。若所选空间的对应结果不可用（例如设了 `umap` 但 `umap-learn` 未装），会打印警告并自动回退到 PCA。
- **模式**：
  - `--mode cg`：分析粗粒化轨迹（读取 `*_cg.lammpstrj`）
  - `--mode aa`：分析全原子轨迹
- **常用 CLI 参数**：
  - `--max-frames N`：限制总帧数
  - `--max-per-file N`：限制每个轨迹文件的帧数（CG 模式）
  - `--cluster-space {pca,tsne,umap}`：覆盖 `analysis_atomic.clustering.space`；`tsne`/`umap` 在对应结果缺失时回退到 `pca`。
- **输出**：
  - `pca_results.csv`, `tsne_results.csv`, `umap_results.csv`, `descriptors.csv`（UMAP 列为 `UMAP1..UMAPn`，`n = analysis_atomic.umap.n_components`）
  - `outlier_structures.csv`
  - `figures/` 目录下的 PNG 可视化图（含 `umap_overall.png` / `umap_<sim>.png`；UMAP 被跳过时不生成）
  - CG 模式下，以上文件统一加 `CG_` 前缀（例如 `CG_umap_results.csv`）。
  - 自 v2 起，上述 CSV 末尾统一附带溯源列 `structure_id, source_file, temp`，其中 `structure_id` 形如 `<sim>/<temp|ramp>@<timestep>`，便于从 PCA/t-SNE/UMAP/聚类空间的点回到原始 dump 帧。

#### 4.6.1 降维方法对比与聚类空间选择

| 方法 | 类型 | 保留结构 | 计算成本 | 默认维度 | 推荐用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PCA**    | 线性   | 全局（方差最大）         | 低（一次 SVD）              | 10（取前 3 维聚类） | 基线、快速、可解释；默认聚类空间 |
| **t-SNE**  | 非线性 | 局部（邻域结构）         | 中（`O(N log N)` 量级）     | 2（仅可视化）       | 二维可视化、发现聚类簇 |
| **UMAP**   | 非线性 | 局部 + 较好保留全局      | 中高（首次建图较慢）        | 5（兼顾聚类+可视化） | 既可用于聚类，又可用于可视化； manifold 假设更强 |

- **何时切换 `clustering.space`**：默认在 PCA 前 3 维上聚类。若 SOAP 描述符高度非线性（典型情形：高温/低温/ramping 跨态样本混合），把 `analysis_atomic.clustering.space` 改为 `umap`（或 CLI 加 `--cluster-space umap`）通常能得到更紧致的簇。t-SNE 的 2 维投影波动较大，**不推荐**用作正式聚类空间，主要留给可视化。
- **UMAP 缺失时的降级**：`HAS_UMAP=False` 时整条流水线照常运行 —— UMAP 步骤、`umap_*.png` 绘图、`umap_results.csv` 写盘均被跳过；若此时 `clustering.space='umap'`，聚类器会打印 `Warning: clustering.space='umap' but no ... UMAP result; falling back to PCA.` 并改用 PCA。
- **复现性**：PCA 与 UMAP 均受 `random_state=42` 控制（UMAP 内部用 `numpy.random.seed`）；t-SNE 也固定 `random_state=42`。同一份描述符跨机器结果一致。
- **`select-structures` 联动**：`umap_results.csv` 可直接喂给 `cgkit select-structures`（`--space umap` 或按列名 `UMAP1..n` 自动识别）。当 CSV 同时含 `PC*`/`UMAP*`/`tSNE*` 列时，自动检测优先级为 PCA > UMAP > t-SNE。

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

### 4.8 `cgkit select-structures` — 基于聚类的结构筛选

- **功能**：消费 `cgkit analyze-atomic` 产出的 `pca_results.csv` / `tsne_results.csv` / `umap_results.csv`（含 `PC1..PC10` / `tSNE1,tSNE2` / `UMAP1..UMAPn` 坐标 + `structure_id, source_file, temp, timestep`），在所选投影空间上重新聚类（KMeans / DBSCAN，参数可调），对每个类用 **maximin（最远点贪心采样）** 选 N 个最具多样性的构型，并把每个被选中的帧从源 dump 文件抽取成独立的 LAMMPS dump 文件复制到指定输出目录。用于从大轨迹库里挑出代表性、覆盖度高的训练子集。
- **对应模块**：`cglib/select_structures.py`（新增）
- **配置节**：`select_structures.{input, output_dir, method, space, n_clusters, min_samples, include_noise, seed}`
- **前置工作流**：先 `cgkit analyze-atomic`（产出 CSV），再 `cgkit select-structures`。这样筛选参数（聚类数、N 等）可反复试，无需重跑昂贵的 SOAP/PCA/t-SNE/UMAP 流程。
- **常用 CLI 参数**：
  - `--input FILE`：输入 CSV（`pca_results.csv` / `tsne_results.csv` / `umap_results.csv` / CG 版本，来自 `analyze-atomic`）
  - `--output-dir DIR`：输出目录（选中帧 dump + manifest）
  - `--n N`（必填）：每个类选 N 个构型
  - `--space {pca,tsne,umap}`：投影空间（省略则按列名 `PC1` / `UMAP1` / `tSNE1` 自动检测；同时存在多类列时优先级 PCA > UMAP > t-SNE）
  - `--method {kmeans,dbscan}`：聚类算法（默认 `kmeans`）
  - `--n-clusters K`：KMeans 簇数（默认 8；自动 cap 到 `len(X)//2`）
  - `--eps FLOAT`：DBSCAN 邻域半径（省略则取成对距离 30 百分位）
  - `--min-samples N`：DBSCAN 核心点最小邻居数（默认 5）
  - `--include-noise`：把 DBSCAN 噪声点（label -1）当作可采样簇（默认跳过）
  - `--seed N`：KMeans 随机种子（默认 42）

- **maximin 选择算法**：每个簇内，第一个选中点是离簇质心最近的点（确定性，无需 RNG）；之后每一步选使「到已选点集的最小距离」最大的未选点（经典最远点采样 / FPS）。这样保证选出的 N 个构型在投影空间里彼此尽量分散，最大化结构覆盖度。簇内点数 < N 时全选，并打印 `[small-cluster]` 告警。

- **帧抽取规则**：按 `source_file` 分组选中行，每个源 dump 至多解析一次；用 `timestep` 列（缺失时从 `structure_id` 末段 `@<ts>` 解析，适配 CG CSV）定位帧，调用 `cglib.lammps.write_lammps_frame` 写成单帧 dump。timestep 未命中则打印 `[no-frame]` 跳过。

- **输出**（位于 `<output_dir>/`）：
  - 每个选中帧一个 `.lammpstrj`：文件名 = `structure_id` 净化后（`/`、`@` → `_`），如 `1-npt_200_100000.lammpstrj`，含该 timestep 的 `id type xu yu zu c_pe fx fy fz`（与原 dump 同列）。
  - `selection_manifest.csv`：列为 `structure_id, cluster, selection_rank, source_file, timestep, temp, <PC1..PCk 或 tSNE1..tSNEk>, output_file`，记录全部选择结果便于溯源。

- **典型用法**：
  ```bash
  # 先跑分析（一次性产出 pca_results.csv）
  cgkit analyze-atomic --mode aa --max-frames 2000

  # 反复试不同聚类参数，无需重跑分析
  cgkit select-structures --input pca_results.csv --output-dir sel_k8n3 \
      --n 3 --method kmeans --n-clusters 8
  cgkit select-structures --input pca_results.csv --output-dir sel_tsne \
      --n 5 --space tsne        # 自动用 tSNE 列聚类
  ```

- **边界情况**：DBSCAN 把全部点判为噪声且未加 `--include-noise` 时报错退出；CG CSV 无 `timestep` 列时从 `structure_id` 解析；源 dump 找不到对应 timestep 时该行 `output_file` 留空并告警。

---

### 4.9 `cgkit cg-verify` — 粗粒化结果校核

- **功能**：把 CG 粒子 CSV（`*_particles.csv`）与对应的源原子 LAMMPS dump 逐帧比对，自动揪出粗粒化过程中的常见 bug，尤其是「使用 wrapped 坐标做最近邻匹配，导致同一 CG 粒子的成员原子横跨周期边界」这类隐蔽错误。

- **对应模块**：`cglib/cg_verify.py`（新增）

- **两种模式**：

  - **`--mode auto`（默认）**：对每个 CG CSV 跑 4 项检查
    1. **`pbc`（跨周期边界检测）**：对每个非手工分配的 CG 粒子，计算其成员原子在 x/y/z 三轴上的坐标跨度（最大值 - 最小值），除以盒子边长得到分数；超过 `--pbc-thresh`（默认 0.45，即 45%）报 FAIL，超过一半（22.5%）报 WARN。手工分配（`manual_assignment=True`）的粒子允许故意跨界，跳过。
    2. **`conservation`（守恒量重算）**：精确镜像 `cg_gen.create_cg_particle` 的算法 —— 位置取中心原子（`atom_indices` 首个索引）、力/PE 在 `average_*` 开启时取成员均值 —— 用原子数据重算后与 CSV 存储值对比。容差：力 `--force-tol`（默认 `1e-4` eV/Å），PE `--pe-tol`（默认 `1e-6` eV），位置必须严格相等（误差 > `1e-9` 即 FAIL）。
    3. **`coverage`（原子覆盖性）**：每个原子行索引必须**恰好**出现在一个 CG 粒子的 `atom_indices` 里。遗漏或重复都报 FAIL。
    4. **`manual`（手工分配保真）**：config 的 `coarse_graining.cg_assignments` 每一条都必须对应 CSV 里**恰好一行** `manual_assignment=True`、`assigned_atom_ids` 匹配、CG type 匹配的记录；缺失/重复/类型不符都报 FAIL，CSV 里多出来的未声明手工分配报 WARN。

  - **`--mode manual --atoms ID1 ID2 ...`**：对每个用户指定的原子 ID，报告它属于哪个 CG 粒子、同粒子的兄弟原子 ID、CG 位置、该原子自己的位置、以及 quick PBC 跨界标志。用于追踪 `--mode auto` 报出的具体原子。

- **文件发现**：
  - 默认：扫描 `paths.cg_data_base_dir/<sim>/` 下排序后的第一个 `*_particles.csv`（先经 `--sim`/`--temp` 过滤）
  - `--all`：扫描所有 enabled sim 的全部文件
  - `--file PATH`：显式指定单文件（与 `--all` 互斥）
  - `--max-files N`：给 `--all` 加上限

- **原子源解析**：从 CG 文件名剥掉 `_particles.csv` 后缀（如 `foo.lammpstrj_particles.csv` → `foo.lammpstrj`），在 `paths.aa_data_base_dir`（可用 `--atomic-dir` 覆盖）下按 sim 的 `trajectory_dir`/`data_subdir`/`output_subdir` 逐级查找；都找不到则递归兜底。

- **配置节**：`verify_cg.{output_dir, checks, force_tolerance, pe_tolerance, pbc_span_threshold}`、`paths.{cg,aa}_data_base_dir`、`coarse_graining.{position_source, average_forces, average_potential_energy, cg_assignments}`、`simulations`。

- **常用 CLI 参数**：
  - `--mode {auto,manual}`：auto = 全套 4 项检查（默认）；manual = 查询 `--atoms` 列表
  - `--atoms ID [ID...]`：要查询的原子 ID（仅 manual 模式有效）
  - `--file PATH | --all`：单文件（互斥）| 全量扫描
  - `--max-files N`：限制 `--all` 的扫描文件数
  - `--base-dir DIR`：覆盖 `paths.cg_data_base_dir`（CG CSV 根）
  - `--atomic-dir DIR`：覆盖 `paths.aa_data_base_dir`（原子 dump 根）
  - `--output-dir DIR`：报告 CSV 输出目录
  - `--checks pbc conservation coverage manual`：选子集（默认全选）
  - `--force-tol TOL`：力重算容差，eV/Å（默认 `1e-4`）
  - `--pe-tol TOL`：PE 重算容差，eV（默认 `1e-6`）
  - `--pbc-thresh FRAC`：PBC 跨界 FAIL 阈值，盒子长度分数（默认 `0.45`）
  - `--no-csv`：不写 `cg_verify_report.csv`
  - `--failures-only`：CSV 只写 FAIL 行
  - `--quiet` / `-q`：只在该文件有 FAIL 时才打印详细信息
  - `--sim` / `--temp` / `--workers`：通用过滤/并行参数

- **输出**：
  - **stdout**：逐文件人类可读报告（逐帧 FAIL/WARN 计数、最大重算误差）+ 最终汇总。
  - **`<output_dir>/cg_verify_report.csv`**（仅在有问题时生成）：每行一个 issue，列为 `file, sim, temp, timestep, check, severity, cg_id, message, n_atoms, member_atom_ids, force_err, pe_err, pos_err, pbc_span_frac_{x,y,z}`。

- **退出码**：
  - `0`：全部通过（允许 WARN）
  - `1`：至少一个 FAIL
  - `2`：文件级错误（原子源缺失 / 解析失败 / timestep 不匹配），无法完成校核
  - `3`：CLI 用法错误（如 `--mode manual` 忘带 `--atoms`）

- **典型用法**：
  ```bash
  # 默认单文件 auto 模式（拿第一个 CSV 试水）
  cgkit cg-verify

  # 指定原子源目录（CG 来自选择性采样时常用）
  cgkit cg-verify --atomic-dir /path/to/slim_atomic_data

  # 全量扫描 + 限制 20 个文件 + 静默
  cgkit cg-verify --all --max-files 20 --quiet

  # 只跑 PBC 检查、收紧阈值到 30%
  cgkit cg-verify --checks pbc --pbc-thresh 0.30

  # 手动查询 config.cg_assignments 里的 8 个原子
  cgkit cg-verify --mode manual --atoms 1 67 68 69 66 197 198 199

  # 显式指定文件 + 只输出 FAIL 到 CSV
  cgkit cg-verify --file /path/to/foo_particles.csv --failures-only
  ```

- **边界情况**：
  - 原子文件缺失 → worker 返 `(False, "atomic source not found", {})`，退出码 2
  - 三斜盒（tilt 非零）→ 当前 `LammpsDumpReader` 会丢弃 tilt；stdout 打一条 INFO 警告，检查继续
  - `position_source="unwrapped"` 但 dump 无 `xu/yu/zu` → 复用 `cg_gen.py` 的回退逻辑（用 wrapped），stdout 打印 INFO
  - `cg_assignments` 为空 → `verify_manual_fidelity` 返回空列表，不报错
  - 手动模式原子 ID 不在文件 → 打印 "NOT FOUND"，继续处理其它 ID
  - CSV 多 timestep → 按 timestep 分组，每组分别跑 4 项检查，原子文件用匹配 timestep 的 frame
---

### 4.10 `cgkit plot-test` — DeepMD 模型预测对比

- **功能**：读取 DeepMD 模型预测输出（`dp test` 生成）与系统参考数据，生成 parity 对比图与误差分布直方图，输出 RMSE/MAE/R² 汇总指标。用于训练完成后验证 CG 势能的预测精度。

- **对应模块**：`cglib/plot_test.py`（新增）

- **生成的图表**：
  - **Force parity plot**（`force_parity.png`）：Fx/Fy/Fz 三个分量分别绘制预测值 vs 参考值散点图，含 y=x 参考线及 RMSE/MAE/R² 标注。
  - **Energy parity plot**（`energy_parity.png`）：能量预测值 vs 参考值散点图。
  - **Force error distribution**（`force_error_dist.png`）：力误差（F_pred - F_ref）的直方图，标注均值、标准差、RMSE。
  - **Energy error distribution**（`energy_error_dist.png`）：能量误差的直方图。

- **数据格式**：自动支持 `.raw` 和 `.npy` 两种 DeepMD 输出格式；`--ref-dir` 支持自动检测（在 `--pred-dir` 的父目录/兄弟目录中查找）。

- **配置节**：`plot_test.{pred_dir, ref_dir, output_dir, max_frames, skip_plots}`，`paths.{deepmd_output_base_dir, analysis_output_base_dir}`。

- **常用 CLI 参数**：
  - `--pred-dir PATH`：模型预测目录（含 `energy.raw/.npy`、`force.raw/.npy`）
  - `--ref-dir PATH`：参考数据目录（可自动检测）
  - `--output-dir PATH`：输出目录（写入图上和 `test_metrics.csv`）
  - `--max-frames N`：限制对比帧数（均匀降采样）
  - `--skip {force,energy}`：跳过指定类型的图表
  - `--sim` / `--temp` / `--workers`：通用过滤/并行参数

- **输出**（位于 `<output_dir>/`）：
  - `force_parity.png`、`force_error_dist.png`
  - `energy_parity.png`、`energy_error_dist.png`
  - `test_metrics.csv`：列为 `quantity, rmse, mae, r2, n_frames`

- **典型用法**：
  ```bash
  # 基本用法
  cgkit plot-test --pred-dir test_output/ --ref-dir system_data/

  # 限制帧数 + 只比较力
  cgkit plot-test --pred-dir test_output/ --ref-dir system_data/ \
      --max-frames 100 --skip energy
  ```

---

### 4.11 `cg-gen` 算法升级：PBC 链式 unwrap + r_cutoff + id_pattern

`cg-gen` 在原两阶段算法（`cg_assignments` → `patterns`）基础上，新增三项
能力：周期性边界 unwrap 预处理、`r_cutoff` 距离截断、`id_patterns` 基于
id 偏移的明确绑定。三者均为可选——不写新 config 字段时行为完全等同旧版。

#### 4.10.1 三阶段优先级（升级后）

```
Step 1   cg_assignments  （显式 atom_ids，最高优先级）
Step 1.5 id_patterns     （type_pattern + id_offsets，新增）
Step 2   patterns        （距离最近邻，兜底）
```

任一阶段消费的原子都会从后续阶段的候选池里剔除，因此一个原子只会被一个
bead 占有。

#### 4.10.2 PBC 链式 unwrap 预处理

**触发**：`coarse_graining.unwrap_pbc=true`（默认值；CLI 用 `--unwrap-pbc` /
`--no-unwrap-pbc` 覆盖）。`unwrap_method` 当前仅支持 `"chain"`（保留为扩展位）。

**算法**：对每帧原子坐标，**按 id 顺序**对每个原子相对前一个原子做最小镜像
折叠：

```python
for i in 1..N:
    d = coord[i] - coord[i-1]
    coord[i] -= round(d / L) * L        # L = box length on that axis
```

折叠后的坐标写入 `x/y/z` 列并同步到 `xu/yu/zu`，后续的距离匹配、位置存储、
CG dump 输出全部基于这些"物理实际坐标"。

**物理含义**：wrapped 坐标在原子穿越周期边界时会发生 ±L 的跳变。两条链分别
位于盒子两端时，其 wrapped 坐标相距 ~L，但物理上可能近在咫尺。链式 unwrap
按 id 顺序把同一条链的原子"拼回"连续轨迹，消除这种伪距。

**适用前提**：链式 unwrap 假设 **id 顺序 = 化学键顺序**（即 dump 按分子 /
链内原子顺序输出）。这是大多数 LAMMPS `write_data` 默认行为的成立条件。

> ⚠ **不适用场景**：若 dump 把所有 type 1（C）列在前面、所有 type 2（H）
> 列在后面（如本工作流 `01.aa_small` 的某些 dump），则 C→H 边界处（如
> `id=66` 的末位 C 到 `id=67` 的首位 H）会按非化学键顺序做折叠，把 H 错误
> 地"拉"到 C 链末端附近。这种 dump 应使用 `--no-unwrap-pbc` 关闭 unwrap，
> 或先用 `id_patterns` 明确指定 C-H 绑定关系。

#### 4.10.3 `r_cutoff` 距离截断

**触发**：`coarse_graining.r_cutoff` 为数值（默认 1.25 Å；`null` 表示不限）。
CLI 用 `--r-cutoff` 覆盖。

**算法**：Step 2 (patterns) 阶段，在升序排序候选 H 之前，先剔除到中心原子
平方距离 > `r_cutoff²` 的候选。若截断内候选数 < pattern 所需 H 数，该中心
原子无法形成 bead，会被跳过并触发警告：

```
WARNINGS (r_cutoff):
  中心原子 id=X 无匹配 pattern (r_cutoff=1.25)  (×N 次)
```

**用途**：当多个 C 竞争同一个 H 时，距离远的 C 不应"偷"走近邻 C 的 H。
1.25 Å 约为 C-H 键长的 1.2 倍，足以容纳振动带来的距离浮动。

#### 4.10.4 `id_patterns`（id 偏移绑定）

**触发**：`coarse_graining.id_patterns` 列表非空。

**格式**：

```json
"id_patterns": [
  {
    "type_pattern": [1, 2, 2],          // 与 patterns 同语义
    "id_offsets":   [0, 67, 68],        // 第 i 个成员 = center_id + offset
    "cg_type":      1,                  // 该 bead 的 CG type
    "description":  "C + 2 H by id offset"
  }
]
```

**约束**：

| 校验 | 失败处理 |
| :--- | :--- |
| `len(type_pattern) == len(id_offsets)` 且 ≥ 2 | 跳过该规则，警告 |
| `id_offsets[0] == 0`（中心必须是自身） | 跳过该规则，警告 |
| `type_pattern[0] == center_atom_type` | 不是当前 center_type 的规则，跳过（不警告） |

**匹配流程**（对每个未被占用的中心原子）：

1. 按 `center_id + id_offsets[i]` 查找每个成员原子的 id；
2. 任一成员 id 不存在 / type 不匹配 / 已被占用 → 该中心原子跳过此规则，
   警告消息按 `id_pattern_id_missing` / `id_pattern_type_mismatch` /
   `id_pattern_atom_used` 分类，留给后续 patterns 阶段兜底；
3. 全部通过 → 创建 bead，`match_status=id_pattern`，原子标记为已占用。

**适用场景**：dump 的 id 编号有统一规律时（例如"每个 CH₂ 的两个 H 的 id
正好是 C id +1, +2"），用 `id_patterns` 比依赖距离更稳健、更快（无 O(N²)
邻居搜索）。

> ⚠ **不适用场景**：若 id 编号无统一偏移规律（例如本工作流 `01.aa_small`
> 的 PE dump：所有 C 在 id 1–66，所有 H 在 id 67–200，H 的偏移因 C 索引
> 而异：C1→+67/+68、C2→+69/+70、…），常量偏移会大面积冲突，警告刷屏。这种
> dump 应使用默认 `patterns`（距离匹配），不用 `id_patterns`。

#### 4.10.5 `match_status` 列

每个 CG bead 在 `_particles.csv` 里的 `match_status` 字段记录其来源阶段：

| 值 | 来源 | 其它相关列 |
| :--- | :--- | :--- |
| `manual` | Step 1 `cg_assignments` | `manual_assignment=True`, `assigned_atom_ids` |
| `id_pattern` | Step 1.5 `id_patterns`（新增） | `id_pattern` 列含命中规则 |
| `pattern` | Step 2 `patterns`（距离） | `pattern` 列含匹配的 type 列表 |

#### 4.10.6 警告汇总

`cg-gen` 运行末尾打印去重计数后的警告（避免百万次刷屏），形如：

```
WARNINGS (id_pattern):
  id_pattern: 中心 id=2 缺 id=70                                (×32 次)
  id_pattern: 中心 id=3 缺 id=71                                (×32 次)
  ...
WARNINGS (r_cutoff):
  中心原子 id=5 无匹配 pattern (r_cutoff=1.25)                   (×101 次)
```

按 `(类别, 消息)` 去重，括号内是相同消息的累计次数。`id_pattern_*` 类
警告表示该中心原子已留给后续 patterns 阶段兜底，**不一定是错误**——只有
当 `patterns` 阶段也兜底失败（`r_cutoff` 警告）时才说明真的漏配。

#### 4.10.7 典型配置范例

**A. 关闭所有新特性（最严格的旧版兼容）**：

```json
"coarse_graining": {
  "patterns": [[1, 2, 2]],
  "r_cutoff": null,
  "unwrap_pbc": false,
  "id_patterns": []
}
```

**B. 启用 unwrap + r_cutoff，无 id_pattern（推荐基线）**：

```json
"coarse_graining": {
  "patterns": [[1, 2, 2]],
  "r_cutoff": 1.25,
  "unwrap_pbc": true,
  "unwrap_method": "chain",
  "id_patterns": []
}
```

**C. 全部启用（dump 的 id 编号规律一致时）**：

```json
"coarse_graining": {
  "patterns": [[1, 2, 2]],
  "r_cutoff": 1.25,
  "unwrap_pbc": true,
  "unwrap_method": "chain",
  "id_patterns": [
    {
      "type_pattern": [1, 2, 2],
      "id_offsets":   [0, 1, 2],
      "cg_type":      1,
      "description":  "CH2: C_id, C_id+1, C_id+2"
    }
  ]
}
```

#### 4.10.8 边界情况

| 情况 | 处理 |
| :--- | :--- |
| dump 无 `id` 列 | unwrap 按行号顺序；id_pattern 跳过并 warning |
| `id_offsets[0] != 0` | 跳过此 id_pattern，warning |
| `len(type_pattern) != len(id_offsets)` | 跳过此 id_pattern，warning |
| id_pattern 候选 id 不存在 | warning，中心原子留给 patterns |
| id_pattern 候选 type 不匹配 | warning，同上 |
| id_pattern 候选已被占用 | warning，同上 |
| `cg_assignments` 与 `id_pattern` 抢同一原子 | `cg_assignments` 先占用，`id_pattern` 报 `atom_used` |
| `r_cutoff` 过小导致全部 pattern 不匹配 | bead 不生成，stdout 末尾汇总 |
| unwrap 后坐标远超原盒子 | 正常（CG bead 也用 unwrap 坐标）；OVITO 可视化需注意 |
| 旧 config 无新字段 | 用代码默认值，行为完全等同旧版 |

#### 4.10.9 与 `cg-verify` 的协同

`cg-verify` 已同步升级：

- 读 `coarse_graining.{unwrap_pbc, r_cutoff, id_patterns}` 用于决定原子数据
  是否走 unwrap 流程；
- `verify_pbc_span` 与 `verify_conservation` 在 `unwrap_pbc=true` 时对原子
  DataFrame 也跑一次 `_unwrap_chain_coords`，与 cg-gen 保持同一参考系；
- `verify_conservation` 的位置比较改为最小镜像差
  `|d - round(d/L)*L|`，无论 CG CSV 存的是 wrapped 还是 unwrap 后坐标，
  都能在半个盒长内正确比较。

预期：启用 unwrap 后，`verify_pbc_span` 报告的 FAIL 数应大幅下降（理想情况
降到 0）。若仍出现 FAIL，说明 dump 的 id 编号不符合"链式"假设，应改用
`--no-unwrap-pbc` 或改用 `id_patterns` 显式绑定。

---

## 5. 配置文件（`config.json`）

`config.json` 是统一配置中心，包含 12 个顶层配置节：

| 配置节 | 说明 |
| :--- | :--- |
| `paths` | 输入/输出根目录路径（原子数据、CG 数据、DeepMD 数据、日志、分析输出） |
| `simulations` | 模拟列表（如 `1-npt`, `2-nvt`, `3-upT`, `4-dnT`） |
| `coarse_graining` | 粗粒化方法、映射模式、中心原子类型等 |
| `deepmd` | DeepMD 数据格式参数（组数、类型列使用等） |
| `fparam` | 温度单位、extract/const 的模拟名称与温度列表 |
| `analysis_cg` | 采样数、最大文件数、输出目录等 |
| `analysis_atomic` | 分析模式、最大帧数、SOAP/PCA/t-SNE/UMAP/聚类（含 `space` 字段）/GNN 参数 |
| `plot_pt` | P–T 覆盖图的输出目录与最大帧数（`output_dir`, `max_frames`） |
| `select_structures` | 结构筛选的输入 CSV、输出目录、聚类方法/参数（`input`, `output_dir`, `method`, `space`, `n_clusters`, `min_samples`, `include_noise`, `seed`） |
| `plot_test` | 模型预测对比的输入/输出目录、最大帧数、跳过项（`pred_dir`, `ref_dir`, `output_dir`, `max_frames`, `skip_plots`） |
| `verify_cg` | CG 校核的输出目录、检查项、容差（`output_dir`, `checks`, `force_tolerance`, `pe_tolerance`, `pbc_span_threshold`） |
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
