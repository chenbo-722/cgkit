# 可视化增强改动总结

本轮改动针对 `cgkit` 工具集做了两处可视化增强：

1. **Task 1** — `cgkit select-structures` 新增选中构型的可视化输出
   （`selection_overview.png`）。
2. **Task 2** — `cgkit analyze-atomic` 的 `pca_overall.png` 右面板从
   「按 sim_type 分类离散着色」改为「按 **temperature** 离散步阶着色，
   所有 sim 数据汇总到同一面板」。

---

## Task 1 — `select-structures` 出图

### 目标

在筛选构型模块里，把全部点（按 cluster 离散着色作淡色背景）+ 选中的
N×K 个点（醒目红圈覆盖 + selection_rank 标注）画成一张
`selection_overview.png`，与已有的 `selection_manifest.csv`、单帧 dump
文件一起输出。

### 涉及文件

| 文件 | 改动 |
|---|---|
| `cglib/select_structures.py` | 顶部加 `_import_deps()`；新增 `_plot_selection_overview()`；`run()` 末尾加 try/except 调用 |
| `cglib/analyze_cg.py` | 只读参考：`_import_deps` 写法模板 |

### 关键代码

#### (1) 顶部 lazy matplotlib import（lines 37–55）

完全照抄 `cglib/analyze_cg.py` 的 headless-safe 写法：

```python
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
```

#### (2) 新增 `_plot_selection_overview()`（lines 318–415）

签名：

```python
def _plot_selection_overview(df_clean: pd.DataFrame,
                             selected: pd.DataFrame,
                             feat_cols: List[str],
                             method: str,
                             output_dir: str) -> str:
```

算法要点：

1. `_import_deps()`；`len(feat_cols) < 2` 时 print 跳过并 `return ''`。
2. `x_col, y_col = feat_cols[0], feat_cols[1]`（PC1/PC2 或 tSNE1/tSNE2）。
3. 底层散点：
   - noise (-1)：`c='#cccccc', s=10, alpha=0.4, label='Noise'`，zorder=1。
   - 每个真簇：`plt.cm.get_cmap('tab10', 10)`（>10 簇循环），`s=15,
     alpha=0.5`，zorder=2，label=`f'Cluster {c}'`。
4. 高亮选中：通过 `structure_id` 集合在 `df_clean` 上反查 sel_mask
   （回退到 selected.index 位置对齐）。画
   `marker='o', facecolors='none', edgecolors='red', s=90,
   linewidths=1.6, zorder=4`，label=`f'Selected (N={n})'`。
5. rank 标注：仅当 `n_selected <= 30` 时，对每个选中点
   `ax.annotate(str(rank), xy=(x,y), xytext=(4,4),
   textcoords='offset points', fontsize=7, color='darkred')`。
6. 标题：`f'Structure selection overview ({method}, {N} cluster(s),
   {n_selected} selected)'`。
7. 图例：`loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False,
   fontsize=8`。
8. 保存：`<output_dir>/selection_overview.png`，`dpi=300,
   bbox_inches='tight'`；`plt.close(fig)`；return 路径。

#### (3) `run()` 末尾调用（lines 496–501）

```python
# 8. Overview figure — best-effort, never fatal.
try:
    _plot_selection_overview(df_clean, selected, feat_cols,
                             method, output_dir)
except Exception as e:
    print(f"[plot] selection_overview failed (non-fatal): {e}")
```

best-effort：画图失败永远不影响已经写出的 CSV / dump 文件。

### 边界情况处理

| 情况 | 处理 |
|---|---|
| DBSCAN 全噪声（label 全 -1） | `_select_per_cluster` 已先报错（`--include-noise=False`）或仅画灰色噪声点+红圈（`--include-noise=True`） |
| 单簇 | n_real=1，正常画 1 个簇色 |
| feat_cols 不足 2 列 | print 跳过，return '' |
| 选中点 >30 | 跳过 annotate，仍画红圈 |
| `structure_id` 缺失 | 回退到 selected.index 位置对齐 |
| matplotlib 未装 | `_import_deps` 抛 ImportError；try/except 兜底 |

### 验证（5 组 smoke test 全部通过）

| 测试 | 输入 | 结果 |
|---|---|---|
| KMeans PCA | pca_results.csv, `--n 3 --n-clusters 4` | 4 clusters × 3 = 12 picks, PNG ✓ |
| t-SNE auto-detect | tsne_results.csv, `--n 3` | 8 clusters × 3 = 24 picks, PNG ✓ |
| DBSCAN + `--include-noise` | pca_results.csv, `--n 2 --eps 5` | 全噪声 graceful fallback, 2 picks, PNG ✓ |
| 多小簇 | pca_results.csv, `--n 4 --n-clusters 20` | `[small-cluster]` 警告正确触发, 58 picks, PNG ✓ |
| >30 选中 | tsne_results.csv, `--n 5 --n-clusters 8` | 40 picks, annotate 跳过路径正确, PNG ✓ |

视觉检查：image inspection 确认 `selection_overview.png` 包含 4 个 cluster
颜色（tab10: blue/orange/green/red）、12 个红色空心圆覆盖、selection_rank
数字标注（0/1/2）、正确的标题/坐标轴/图例。

---

## Task 2 — `pca_overall.png` 右面板改温度离散步阶

### 目标

把 `cgkit analyze-atomic` 的 `pca_overall.png` **右面板**从「按 sim_type
分类离散着色」改成「按 **temperature** 离散步阶着色，所有 sim 的数据汇总
到同一面板」。左面板、per-sim 分支、其它图（`tsne_overall.png`、cluster
图、combined 图）一律不动。

### 涉及文件

| 文件 | 改动 |
|---|---|
| `cglib/analyze_atomic.py` | 第 949–959 行右面板代码整体替换为温度离散步阶 scatter |

### 关键代码

#### 替换前（lines 949–959，按 sim_type 分类）

```python
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
_apply_lego_legend(ax, unique_sim_types, outside=len(unique_sim_types) > 6)
```

#### 替换后（lines 949–969，按温度离散步阶）

```python
ax = axes[1]
unique_temps = sorted(list(set(temperatures)))
if len(unique_temps) >= 2:
    bounds = np.linspace(min(unique_temps) - 25,
                         max(unique_temps) + 25, 11)
    n_colors = 10
else:
    # Single temperature across the whole dataset.
    t0 = float(unique_temps[0])
    bounds = np.array([t0 - 50, t0 + 50])
    n_colors = 1
temp_norm_disc = BoundaryNorm(bounds, ncolors=n_colors)
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                     c=temperatures, cmap=plt.cm.RdBu_r,
                     norm=temp_norm_disc, alpha=0.6, s=30)
ax.set_xlabel('PC1 (%.1f%%)' % get_var(0))
ax.set_ylabel('PC2 (%.1f%%)' % get_var(1))
ax.set_title('Overall: PC1 vs PC2 (colored by temperature)')
cbar = plt.colorbar(scatter, ax=ax, label='Temperature (K)')
if 1 <= len(unique_temps) <= 10:
    cbar.set_ticks(unique_temps)
```

`BoundaryNorm` 已通过 `analyze_atomic.py` 的 `_import_deps()` 导入
（`from matplotlib.colors import BoundaryNorm`），无需新加 import。

### 算法说明

- **多温度档（≥2）**：`bounds = np.linspace(min-25, max+25, 11)` 生成 11
  个边界 → 10 个离散 bin；`BoundaryNorm(bounds, ncolors=10)` 把每个温度
  值映射到对应 bin；`cmap=RdBu_r` 给 10 种颜色。
- **单温度档**：bounds=[t0-50, t0+50]，n_colors=1，单色显示（不崩）。
- **colorbar**：1–10 个温度档时 `cbar.set_ticks(unique_temps)` 显示真实
  档位刻度（如 200/300/400/500/600）；>10 档（ramping 帧）用默认刻度。

### 保持不变

- 左面板（axes[0]，940–947 行）：连续 RdBu_r 能量色阶。
- per-sim PCA 分支（970–1013 行）：原来的 per-sim 温度着色逻辑。
- `tsne_overall.png`、t-SNE per-sim、`cluster_*`、`combined_*`。
- `_apply_lego_legend` 和 `_categorical_palette`：保留在文件中，仍被
  t-SNE overall、per-sim 等其它地方调用。

### 边界情况

| 情况 | 处理 |
|---|---|
| 所有温度相同（如只跑了 200K） | `len(unique_temps)==1` 分支：bounds=[t0-50, t0+50]，n_colors=1，单色 |
| 温度档位 ≤10（如 5 个：200/300/400/500/600） | `cbar.set_ticks(unique_temps)` 显示真实档位刻度 |
| 温度档位 ≥11（含 ramping 帧的连续值） | 跳过 set_ticks，用 colorbar 默认刻度 |
| `temperatures` 缺失 | `_extract_metadata` 已用 300 兜底，不会缺 |

### 验证

```bash
cgkit analyze-atomic --mode aa --max-frames 60 --temp 200 300 400 500 600 \
    --output-dir /tmp/aa_smoke
```

输出确认：

- `/tmp/aa_smoke/figures/pca_overall.png`（262 KB）重新生成
- 右面板标题 `colored by temperature`，RdBu_r 离散步阶
- colorbar 显示 200/300/400/500/600 真实档位
- 左面板、per-sim 图、`tsne_overall.png` 全部保持不变（regression check）

---

## Follow-up（不在本轮）

- 同样改 `tsne_overall.png` 右面板（用户未指定，本轮不动）。
- 让 `analyze-atomic` 的 pca/tsne CSV 附带 cluster 列，让 select-structures
  可直接沿用已有聚类（避免重跑）。
- `select-structures` 可选 `--no-plot` 标志（headless 场景跳过画图）。
- `selection_overview.png` 中 tab10 的第 4 个颜色（红）与选中红圈存在轻微
  视觉竞争，可考虑后续换成 dark red / black 边框。
