# cgkit 结构分析算法技术报告

## 基于 SOAP-PCA-t-SNE-聚类的高维构型表征与筛选

---

## 1. 摘要与项目背景

本文档详细阐述 `cgkit analyze-atomic` 子命令中结构分析算法的完整技术实现，涵盖从原始 LAMMPS 轨迹到可解释低维投影、再到聚类驱动的代表性构型筛选的全过程。该流程面向聚乙烯（PE）分子体系，支持对**全原子（AA）**和**粗粒化（CG）**两种模式的轨迹进行分析，核心目标是通过高维结构指纹的降维与聚类，量化构型多样性、识别结构族群，并为 DeepMD-kit 训练集提供高覆盖度、低冗余的代表性子集。

**关键设计决策**：
- 降维与聚类在 **PCA 前 3 维空间**执行，而非原始 SOAP 高维空间或 t-SNE 空间，兼顾计算效率与密度估计的可靠性。
- 描述符计算后统一进行 **StandardScaler 标准化**，消除量纲差异。
- 所有输出 CSV 附带 **溯源列**（`structure_id`, `source_file`, `temp`），支持从投影空间的任意点回溯到原始 LAMMPS dump 帧。
- 重型依赖（`scikit-learn`, `matplotlib`, `dscribe`, `torch`）采用**延迟加载**，保证核心模块的轻量导入。

---

## 2. 整体流程架构

结构分析遵循**"描述 → 降维 → 可视化 → 聚类 → 筛选"**的五阶段流水线：

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────────┐
│  LAMMPS 轨迹    │───→│  SOAP 描述符  │───→│  PCA 降维   │───→│  t-SNE   │───→│  聚类/筛选  │
│  (AA / CG)     │    │  (高维指纹)   │    │ (线性压缩)  │    │(非线性可视化)│    │(DBSCAN/KMeans)│
└─────────────────┘    └──────────────┘    └─────────────┘    └──────────┘    └─────────────┘
         │                    │                  │                │                │
         │                    │                  │                │                └─→ select-structures
         │                    │                  │                └─→ tsne_results.csv
         │                    │                  └─→ pca_results.csv
         │                    └─→ descriptors.csv
         └─→ structure_id, source_file, temp (溯源列)
```

### 2.1 数据加载策略

| 模式 | 输入文件 | 采样策略 | 元数据提取 |
|:---|:---|:---|:---|
| **AA** | `*.lammpstrj`（原始原子 dump） | 每文件取**首帧** | 从路径解析 `sim_type`, `temp`；从文件头读取 `timestep` |
| **CG** | `*_cg.lammpstrj`（粗粒化 dump） | 每文件取**前 `max_per_file` 帧** | 同上；额外从 sibling `_particles.csv` 读取总能量 |

路径解析支持三种项目布局：
1. `<sim>/traj/<file>` — 标准 NPT/NVT 轨迹
2. `<sim>/<temp>/<file>` — 按温度分目录
3. `<sim>/<PREFIX>.<T>.<step>` — 温度嵌入文件名

变温模拟（`3-upT`, `4-dnT`）的 `temp` 标记为 `None`，在 `structure_id` 中渲染为 `ramp`。

---

## 3. SOAP 描述符：局部环境的旋转-平移不变指纹

### 3.1 数学原理

**SOAP（Smooth Overlap of Atomic Positions）** 描述符由 Bartók 等人提出，核心思想是将原子 $i$ 的局部环境 $\mathcal{N}_i$ 表示为一组以 $i$ 为中心的高斯型原子密度与一组正交基函数的内积：

$$
\rho_i(\mathbf{r}) = \sum_{j \in \mathcal{N}_i} \exp\left(-\frac{(\mathbf{r} - \mathbf{r}_{ij})^2}{2\sigma^2}\right) f_{\text{cut}}(r_{ij})
$$

其中 $f_{\text{cut}}$ 为截断函数（cosine cutoff），$\sigma$ 控制高斯宽度。该密度被投影到径向基函数 $g_n(r)$ 和球谐函数 $Y_{lm}(\hat{\mathbf{r}})$ 构成的正交基上：

$$
\langle \rho_i \mid g_n Y_{lm} \rangle = \int \rho_i(\mathbf{r}) \, g_n(r) \, Y_{lm}^*(\hat{\mathbf{r}}) \, d\mathbf{r}
$$

SOAP 的**幂谱（power spectrum）**通过将系数的外积对 $m$ 求和得到旋转不变量：

$$
p_{n n' l}^i = \sum_m \langle \rho_i \mid g_n Y_{lm} \rangle^* \langle \rho_i \mid g_{n'} Y_{lm} \rangle
$$

最终，SOAP 向量是所有 $(n, n', l)$ 组合的扁平化结果，天然满足：
- **平移不变性**：以中心原子为坐标原点
- **旋转不变性**：球谐函数的耦合消除了方向依赖
- **置换不变性**：邻居原子的求和顺序无关

### 3.2 实现细节

**主实现**依赖 `dscribe` 库（`dscribe.descriptors.SOAP`），通过 ASE 的 `Atoms` 对象传递结构信息：

```python
soap = SOAP(
    species=[1, 2],      # 原子类型：1=C, 2=H
    rcut=5.0,            # 截断半径 (Å)
    n_max=8,             # 径向基函数数量
    l_max=6,             # 角动量量子数上限
    sigma=0.5,           # 高斯展宽 (Å)
    periodic=True,       # 启用周期性边界
    average='inner',     # 对所有原子的 SOAP 取平均，得到体系级描述符
)
```

**Fallback 机制**：当 `dscribe` 不可用时，自动回退到简化描述符：

```python
def compute_rotation_invariant_features(positions, types, box):
    # 1. 去心化（平移不变）
    rel_pos = positions - positions.mean(axis=0)
    # 2. 取前 100 个原子对（截断邻居数，避免计算爆炸）
    for i in range(n_atoms):
        for j in range(i+1, min(i+100, n_atoms)):
            dr = rel_pos[i] - rel_pos[j]
            dr -= round(dr / box_lengths) * box_lengths  # 最小镜像 PBC
            dist = norm(dr)
            features.append([dist, types[i], types[j]])
    return np.array(features).flatten()[:500]  # 截断至 500 维
```

该 fallback 保留了**距离+类型**的旋转不变信息，但丢失了完整的径向-角向耦合，精度低于 SOAP，仅用于保证可用性。

### 3.3 标准化

描述符矩阵 $X \in \mathbb{R}^{N \times D}$ 在降维前经过 `StandardScaler` 处理：

$$
X'_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}
$$

其中 $\mu_j$ 和 $\sigma_j$ 分别为第 $j$ 维的均值和标准差。这消除了不同 SOAP 分量间的尺度差异，对 PCA 和聚类的距离度量至关重要。

---

## 4. PCA：线性降维与方差压缩

### 4.1 算法原理

主成分分析（PCA）通过**正交线性变换**将数据投影到方差最大的方向：

$$
\mathbf{PC}_k = \mathbf{X}' \mathbf{v}_k, \quad \text{其中 } \mathbf{v}_k = \arg\max_{\|\mathbf{v}\|=1} \text{Var}(\mathbf{X}' \mathbf{v})
$$

等价于对标准化后数据的协方差矩阵 $C = \frac{1}{N-1} X'^T X'$ 进行特征值分解：

$$
C \mathbf{v}_k = \lambda_k \mathbf{v}_k, \quad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D
$$

### 4.2 实现参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `n_components` | 10 | 保留的主成分数，输出 `PC1..PC10` |
| 输入数据 | 标准化 SOAP 描述符 | `StandardScaler().fit_transform(descriptors)` |

**方差解释**：运行时会输出前 3 个主成分的方差贡献率及累计方差，用于判断压缩质量：

```
PCA explained variance ratio: [0.42, 0.28, 0.15, ...]
Total variance explained: 0.85
```

### 4.3 设计决策：为何保留 10 维？

- 聚类仅在 **PC1-PC3** 上执行（3D 空间足够密度估计，且避免高维稀疏）。
- 保留至 PC10 是为了下游的 `select-structures`：用户在筛选时可选择 `pca` 空间，利用更多维度进行更精细的聚类划分。
- 输出 CSV 固定包含 `PC1..PC10`，保证接口稳定性。

---

## 5. t-SNE：非线性降维与可视化

### 5.1 算法原理

t-SNE（t-Distributed Stochastic Neighbor Embedding）通过**保持局部邻域相似性**将高维数据嵌入低维空间（通常 2D）。核心两步：

**Step 1 — 高维空间中的相似性概率**：

对于数据点 $\mathbf{x}_i$，以条件概率定义邻居分布（以高斯核度量）：

$$
p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}
$$

对称化联合概率：$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$。$\sigma_i$ 通过 **perplexity** 参数间接控制：

$$
\text{Perplexity}(P_i) = 2^{H(P_i)}, \quad H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}
$$

**Step 2 — 低维空间中的 t 分布**：

在低维嵌入点 $\mathbf{y}_i$ 上，使用自由度为 1 的 t 分布（Cauchy 核）重定义相似性，以缓解"拥挤问题"（crowding problem）：

$$
q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}
$$

通过梯度下降最小化 KL 散度：$C = KL(P \| Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$

### 5.2 实现参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `n_components` | 2 | 输出维度，固定为 2D 以适配可视化 |
| `perplexity` | 30 | 有效邻居数，典型范围 5-50 |
| `max_iter` | 1000 | 梯度下降迭代次数 |
| `random_state` | 42 | 保证结果可复现 |

### 5.3 与 PCA 的关系

t-SNE 在**原始标准化描述符**上运行，而非 PCA 结果。这是有意设计：
- PCA 捕获**全局线性结构**，t-SNE 捕获**局部非线性流形**。
- 两者提供互补视角：PCA 的 PC1-PC2 散点图展示方差主导方向，t-SNE 图揭示局部聚类结构。
- 在 PE 体系中，温度驱动的构型转变（如链段有序-无序转变）可能在 t-SNE 中形成清晰分离的簇，而在 PCA 中表现为沿主成分的连续渐变。

---

## 6. 聚类分析：DBSCAN 与 KMeans

### 6.1 聚类空间的选择

聚类在 **PCA 结果的前 3 维**（`pca_result[:, :3]`）上执行，而非原始描述符或 t-SNE 输出。理由：

1. **维度可控**：3D 空间的密度估计可靠，避免高维诅咒。
2. **物理可解释**：PC1-PC3 通常对应体系的主要结构变化模式（如密度、取向序）。
3. **计算高效**：$O(N^2)$ 的成对距离计算在 3D 上代价极低。

### 6.2 DBSCAN（默认方法）

**DBSCAN（Density-Based Spatial Clustering of Applications with Noise）** 基于密度可达性定义簇，无需预设簇数，能识别噪声点。

**参数自动选择**：
- `eps`（邻域半径）：取 **PC1-PC3 空间成对距离的第 30 百分位数**。
  ```python
  distances = pdist(pca_result[:, :3])
  eps = np.percentile(distances, 30)
  ```
  这意味着约 30% 的点对彼此在 `eps` 内，确保核心点有足够的局部邻居。
- `min_samples`（核心点最小邻居数）：默认 5（含自身）。

**输出**：标签 `-1` 表示噪声点（outliers），其余为簇编号 `0, 1, 2, ...`。

### 6.3 KMeans（备选方法）

当簇具有近似球形、等密度的几何特征时，KMeans 更高效：
- `n_clusters`：默认 4，但自动限制为 `min(n_clusters, len(data)//2)`，防止簇数超过数据点半数。
- `random_state=42`：保证可复现。

**对比决策矩阵**：

| 场景 | 推荐方法 | 理由 |
|:---|:---|:---|
| 簇形状不规则、密度不均 | DBSCAN | 基于密度，无需预设簇数 |
| 需要固定数量的训练子族群 | KMeans | 用户可控簇数，便于 maximin 采样 |
| 数据中存在明显噪声/离群点 | DBSCAN | 噪声点标记为 -1，不混入簇内 |
| 大规模数据集（>10k 点） | KMeans | $O(N \cdot K \cdot I)$ vs DBSCAN 的 $O(N^2)$ |

### 6.4 异常检测（Outlier Extraction）

分析流程内置三种异常检测策略：

| 方法 | 原理 | 适用场景 |
|:---|:---|:---|
| `zscore` | 对 PC1-PC3 的 Z 分数 > threshold 的点 | 远离主分布的统计离群 |
| `cluster` | DBSCAN 的噪声点（label = -1） | 密度异常的低密度区域 |
| `distance` | 到所有其他点的平均距离 > $\mu + \text{threshold} \cdot \sigma$ | 结构上的极端构型 |

异常结构被提取到 `outlier_structures.csv`，包含完整的溯源信息和投影坐标。

---

## 7. GNN 图嵌入（可选模块）

### 7.1 架构设计

当 `torch` 和 `torch-geometric` 可用时，系统使用一个轻量 3 层 GCN（Graph Convolutional Network）生成图嵌入：

```
Input (node features) → GCNConv(in_dim, 64) → ReLU
                        → GCNConv(64, 64) → ReLU
                        → GCNConv(64, 32) → ReLU
                        → global_mean_pool → Linear(32, 32)
```

**节点特征构造**（16 维）：
- 位 0-1：原子类型 one-hot（C=1, H=2）
- 位 2-15：位置坐标的 sin/cos 编码（周期性编码，每维 3 个谐波）

**边构造**：对每原子取前 20 个邻居（按索引顺序），添加双向边。注意：这不是按空间距离的近邻，而是按**索引顺序的拓扑邻居**，设计上偏向于捕捉链式连接结构。

### 7.2 回退机制

若 PyTorch 不可用，GNN 模块自动回退到 **32 维随机高斯嵌入**（`np.random.randn(32)`），保证分析流程不因缺失深度学习依赖而中断。这牺牲了物理意义，但保留了接口兼容性。

### 7.3 可视化

GNN 嵌入输出 2D 散点图（GNN Dimension 1 vs 2），按能量和模拟类型着色，与 PCA/t-SNE 图并列呈现。图拓扑可视化（3D 结构 + 边连接）使用 NetworkX 的 spring layout，当 `networkx` 可用时激活。

---

## 8. 可视化规范：Nature 期刊风格

所有图表遵循统一的出版级视觉规范，代码硬编码于 `_import_heavy_deps()`：

| 规范项 | 设定值 |
|:---|:---|
| 字体 | Arial sans-serif（回退 DejaVu Sans） |
| 字号 | 10 pt |
| 图幅 | 7.2 × 4.5 in（183 mm 双栏宽） |
| 分辨率 | 120 DPI（显示），300 DPI（保存） |
| 轴线 | 无顶部/右侧 spine，线宽 0.8 |
| 网格 | 关闭 |
| 图例 | 无边框，外置左侧 |
| 连续色图 | `RdBu_r`（红-蓝，能量/温度渐变） |
| 分类色板 | 8 色 Brewer 风格（深蓝 → 浅蓝 → 浅红 → 暗红） |

每张图按 **Overall**（全部数据）和 **Per sim_type**（分模拟类型）两个层级输出，确保在全局趋势与局部特征之间无缝切换。

---

## 9. select-structures：基于聚类的代表性构型筛选

### 9.1 流程解耦设计

`select-structures` 是分析流程的**消费端**，独立运行于 `analyze-atomic` 之后。核心解耦：
- `analyze-atomic` 一次性产出 `pca_results.csv` / `tsne_results.csv`（昂贵步骤）。
- `select-structures` 可反复消费这些 CSV，自由调整聚类参数和采样数，无需重跑 SOAP/PCA。

### 9.2 Maximin（最远点）采样算法

对每个聚类簇，执行**贪心最远点采样（Farthest Point Sampling, FPS）**：

**算法**（簇 $C$ 内含 $M$ 个点，选 $N$ 个）：
1. **种子点**：计算簇质心 $\bar{\mathbf{x}} = \frac{1}{M}\sum_{i \in C} \mathbf{x}_i$，选择距离质心最近的点作为第一个选中点（确定性，无需 RNG）。
2. **迭代扩展**：已选集合 $S$，候选集合 $C \setminus S$。选择使下式最大的点：
   $$
   \mathbf{x}^* = \arg\max_{\mathbf{x} \in C \setminus S} \min_{\mathbf{s} \in S} \|\mathbf{x} - \mathbf{s}\|_2
   $$
3. 重复至 $|S| = \min(N, M)$。

**复杂度**：每簇 $O(N \cdot M^2)$（`cdist` 计算），对典型规模（$M \sim 100$, $N \sim 5$）可忽略。

**物理意义**：在投影空间中，选出的 $N$ 个点彼此间距最大，确保每个结构族群内部的最大覆盖度。若某簇点数 $< N$，则全选并打印 `[small-cluster]` 告警。

### 9.3 帧抽取与溯源

选中结果按 `source_file` 分组，每个源 dump 文件**仅解析一次**（通过 `LammpsDumpReader`），用 `timestep` 列定位帧：
- 输出文件名：`{sanitized_structure_id}.lammpstrj`（如 `1-npt_200_100000.lammpstrj`）
- 附带 `selection_manifest.csv`，记录 `structure_id, cluster, selection_rank, source_file, timestep, temp, PC1..PCk, output_file`

---

## 10. 配置参数总览

### 10.1 `analysis_atomic` 配置节

```json
{
  "analysis_atomic": {
    "mode": "cg",                    // "cg" | "aa"
    "max_frames": 500,               // 总帧数上限
    "max_per_file": 10,              // CG 模式下每文件最大帧数
    "output_dir": ".../structure_analysis",
    "skip_gnn_viz": false,
    "gnn_graph_viz": 6,              // 图拓扑可视化样本数
    "gnn_edge_cutoff": null,         // 图边截断（默认 0.3 × 最小盒长）
    "soap": {
      "rcut": 5.0,                   // 截断半径 (Å)
      "n_max": 8,                    // 径向基数
      "l_max": 6,                    // 角动量上限
      "sigma": 0.5                   // 高斯展宽 (Å)
    },
    "pca": {
      "n_components": 10             // 保留主成分数
    },
    "tsne": {
      "n_components": 2,             // 固定 2D
      "perplexity": 30,              // 有效邻居数
      "max_iter": 1000              // 最大迭代
    },
    "clustering": {
      "method": "dbscan",            // "dbscan" | "kmeans"
      "min_samples": 5,              // DBSCAN 核心点最小邻居
      "n_clusters": 4                // KMeans 簇数（默认）
    }
  }
}
```

### 10.2 `select_structures` 配置节

```json
{
  "select_structures": {
    "input": null,                   // pca_results.csv / tsne_results.csv
    "output_dir": null,
    "method": "kmeans",              // "kmeans" | "dbscan"
    "space": "pca",                  // "pca" | "tsne"
    "n_clusters": 8,
    "min_samples": 5,
    "include_noise": false,          // DBSCAN 噪声点是否可采样
    "seed": 42
  }
}
```

---

## 11. 关键设计决策与算法边界

### 11.1 为何 SOAP 后接 PCA，而非直接用 t-SNE 或 UMAP？

- **PCA 提供可解释的方差结构**：每个 PC 的方差贡献率直接量化其信息量，便于理解哪些结构自由度主导了数据变异。
- **t-SNE 不可复用于新点**：t-SNE 没有显式的映射函数，无法将新构型投影到已有嵌入空间。PCA 的投影矩阵 $\mathbf{V}$ 可以保存并复用。
- **计算稳定性**：PCA 为确定性算法（除符号歧义），t-SNE 依赖梯度下降的随机初始化，尽管 `random_state=42` 固定了伪随机性，但超参数敏感性更高。

### 11.2 为何聚类在 PCA 空间而非 t-SNE 空间？

- **距离失真**：t-SNE 刻意压缩全局距离以突出局部结构，可能导致不相似但处于流形远端的点被拉近。在 t-SNE 空间进行密度聚类会产生伪簇。
- **维度可控性**：PCA 前 3 维通常已捕获 60-90% 的方差，足以反映主要结构差异，而 t-SNE 的 2D 嵌入可能丢失了部分中等尺度的分离信息。

### 11.3 参数敏感性分析

| 参数 | 敏感度 | 调参建议 |
|:---|:---|:---|
| `rcut` (SOAP) | 高 | 与体系特征尺度匹配（PE 中 5Å 约 2-3 个 C-C 键长） |
| `n_max`, `l_max` | 中 | 增大会提升分辨率但增加计算量；`n_max=8, l_max=6` 为典型值 |
| `perplexity` (t-SNE) | 中 | 5-50 范围；较小值突出局部结构，较大值保留全局关系 |
| `eps` (DBSCAN) | 高 | 30 百分位数为经验默认；若簇过碎则降低，若全并为一簇则提高 |
| `min_samples` | 中 | 与数据密度相关；稀疏数据集应降低 |

### 11.4 已知局限

1. **正交盒假设**：SOAP 计算和距离计算均假设正交模拟盒（非对角 tilt 因子为 0）。三斜盒需扩展 PBC 处理。
2. **GNN 边构造非物理**：当前按索引顺序取前 20 个邻居，而非空间近邻。对于长链 PE 这恰好对应链上连接，但对于支化或交联体系不适用。
3. **CG 模式的能量读取**：CG 总能量从 sibling `_particles.csv` 的 `c_pe` 列求和得到，若文件缺失则能量信息为空。
4. **t-SNE 计算开销**：在 >2000 点上 t-SNE 的 $O(N^2)$ 成对距离计算成为瓶颈，建议先用 `max_frames` 限制输入规模。

---

## 12. 总结与最佳实践

### 12.1 标准分析流程

```bash
# 步骤 1：一次性运行结构分析（产出投影 CSV）
cgkit analyze-atomic --mode cg --max-frames 500

# 步骤 2：反复筛选，无需重跑分析
cgkit select-structures --input pca_results.csv --n 3 --method kmeans --n-clusters 8
cgkit select-structures --input tsne_results.csv --n 5 --space tsne --method dbscan
```

### 12.2 诊断检查清单

- [ ] PCA 前 3 维累计方差 > 60%？若不足，提高 `n_max`/`l_max` 或检查轨迹质量。
- [ ] DBSCAN 识别的噪声点比例 < 20%？若过高，检查 `eps` 是否过小或体系是否存在多相共存。
- [ ] t-SNE 图中同温度点是否形成连续流形？若出现离散团块，可能存在相变或模拟不稳定性。
- [ ] `select-structures` 的 selection_overview 图中选中点是否均匀覆盖各簇？若某簇未被选中，增加 `n` 或检查该簇是否过小。

### 12.3 性能基准（参考）

| 规模 | SOAP 计算 | PCA | t-SNE | 总计 |
|:---|:---|:---|:---|:---|
| 500 帧 | ~30s | <1s | ~15s | ~45s |
| 2000 帧 | ~2min | <1s | ~3min | ~5min |

（基于 Intel i7 / 16GB RAM，使用 dscribe 的 SOAP）

---

*本报告基于 cgkit v1.0 源码（`cglib/analyze_atomic.py`, `cglib/select_structures.py`）及 `config.json` 配置撰写。算法实现与 legacy 脚本 `0x-analyze_atomic_structure.py` 保持 1:1 行为一致性。*
