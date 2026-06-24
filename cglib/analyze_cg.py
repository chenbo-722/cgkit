"""analyze-cg domain logic: statistical analysis of CG particle CSVs.

Migrated from legacy ``0x-analyze_cg_data.py``. Algorithm preserved 1:1.

Heavy imports (matplotlib, scipy) are deferred to runtime: the
``_import_deps()`` call inside ``run()`` populates module-level globals
so that ``cgkit.py`` importing this module at startup does NOT pull
matplotlib/scipy. This lets ``cgkit cg-gen`` / ``cgkit to-deepmd`` work
without matplotlib installed.
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Matplotlib / scipy are deferred — populated by _import_deps()
plt = None
gridspec = None
gaussian_kde = None


def _import_deps() -> None:
    """Populate matplotlib + scipy module globals. Called from run()."""
    global plt, gridspec, gaussian_kde
    if plt is not None:
        return
    import matplotlib
    matplotlib.use("Agg")  # headless safety
    import matplotlib.pyplot as _plt
    import matplotlib.gridspec as _gridspec
    from scipy.stats import gaussian_kde as _gaussian_kde
    _plt.style.use('seaborn-v0_8-whitegrid')
    _plt.rcParams['font.size'] = 10
    _plt.rcParams['figure.dpi'] = 100
    _plt.rcParams['savefig.dpi'] = 300
    _plt.rcParams['axes.unicode_minus'] = False
    plt = _plt
    gridspec = _gridspec
    gaussian_kde = _gaussian_kde


class CGDataAnalyzer:
    """Statistical analyzer for coarse-grained particle CSV data.

    1:1 port from legacy 0x-analyze_cg_data.py.
    """

    def __init__(self, base_dir, output_dir=None, config_file=None, config=None):
        self.base_dir = Path(base_dir)

        # Accept either a pre-loaded config dict (preferred) or a config_file path.
        self.config = config
        config_output_dir = None
        if self.config is None and config_file:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json_load(f)

        if self.config is not None:
            paths = self.config.get('paths', {}) or {}
            config_output_dir = (paths.get('analysis_cg_output_dir')
                                 or paths.get('output_base_dir'))

        if output_dir:
            self.output_dir = Path(output_dir)
        elif config_output_dir:
            self.output_dir = Path(config_output_dir)
        else:
            self.output_dir = self.base_dir / "analysis_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_data: Dict[str, Any] = {}
        self.summary_stats: Dict[str, Any] = {}
        self.summary_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def find_particle_files(self):
        pattern = str(self.base_dir / "**/*_particles.csv")
        return sorted(glob.glob(pattern, recursive=True))

    def parse_file_info(self, filepath):
        path = Path(filepath)
        try:
            parts = path.relative_to(self.base_dir).parts
        except ValueError:
            parts = path.parts
        info = {
            'filepath': filepath,
            'filename': path.stem,
            'sim_type': parts[0] if len(parts) > 0 else 'unknown',
            'temperature': None,
        }
        if len(parts) > 1:
            try:
                temp_str = parts[1]
                if temp_str.isdigit():
                    info['temperature'] = int(temp_str)
            except ValueError:
                pass
        for part in path.stem.split('.'):
            if part.isdigit():
                temp = int(part)
                if 100 <= temp <= 1000:
                    info['temperature'] = temp
                    break
        return info

    def load_file(self, filepath):
        try:
            return pd.read_csv(filepath)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Failed to load {filepath}: {exc}")
            return None

    def process_single_file(self, filepath, sample_frac=None):
        df = self.load_file(filepath)
        if df is None or len(df) == 0:
            return None
        if sample_frac is not None and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)

        info = self.parse_file_info(filepath)
        stats: Dict[str, Any] = {
            'sim_type': info['sim_type'],
            'temperature': info['temperature'],
            'filename': info['filename'],
            'n_particles': len(df),
            'n_frames': df['timestep'].nunique() if 'timestep' in df.columns else 1,
        }
        if 'fx' in df.columns:
            f_mag = np.sqrt(df['fx'] ** 2 + df['fy'] ** 2 + df['fz'] ** 2)
            stats.update({
                'force_mean': f_mag.mean(), 'force_std': f_mag.std(),
                'force_min': f_mag.min(),  'force_max': f_mag.max(),
                'force_q25': f_mag.quantile(0.25), 'force_q50': f_mag.quantile(0.50),
                'force_q75': f_mag.quantile(0.75), 'force_q95': f_mag.quantile(0.95),
                'force_q99': f_mag.quantile(0.99),
            })
        if 'c_pe' in df.columns:
            stats.update({
                'pe_mean': df['c_pe'].mean(), 'pe_std': df['c_pe'].std(),
                'pe_min': df['c_pe'].min(),   'pe_max': df['c_pe'].max(),
            })
        return stats

    def analyze_dataset(self, sample_frac=None, max_files=None):
        print("Searching for particle data files...")
        files = self.find_particle_files()
        print(f"Found {len(files)} particle files")
        if max_files:
            files = files[:max_files]
            print(f"Processing first {max_files} files")
        all_stats = []
        for i, filepath in enumerate(files):
            if (i + 1) % 50 == 0:
                print(f"Processing file {i + 1}/{len(files)}...")
            s = self.process_single_file(filepath, sample_frac=sample_frac)
            if s:
                all_stats.append(s)
        self.summary_df = pd.DataFrame(all_stats)
        if len(self.summary_df) == 0:
            return self.summary_df
        print(f"Successfully processed {len(all_stats)} files")
        print(f"Total particles: {self.summary_df['n_particles'].sum():,}")
        print(f"Total frames:    {self.summary_df['n_frames'].sum():,}")
        print("\nDataset breakdown:")
        for sim_type in sorted(self.summary_df['sim_type'].unique()):
            data = self.summary_df[self.summary_df['sim_type'] == sim_type]
            print(f"  {sim_type}: {len(data)} files, {data['n_particles'].sum():,} particles")
        return self.summary_df

    # ------------------------------------------------------------------
    # Plotting — uses module-global plt (populated by _import_deps)
    # ------------------------------------------------------------------
    def plot_configuration_count(self, fig_dir):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Dataset Frame Count Analysis', fontsize=14, fontweight='bold')
        sim_stats = self.summary_df.groupby('sim_type').agg({'n_frames': 'sum'})
        sim_stats = sim_stats.sort_values('n_frames', ascending=False)

        ax = axes[0]
        x = np.arange(len(sim_stats))
        bars = ax.bar(x, sim_stats['n_frames'], alpha=0.8, edgecolor='black')
        for bar, count in zip(bars, sim_stats['n_frames']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Simulation Type'); ax.set_ylabel('Frame Count')
        ax.set_title('Frame Count by Simulation Type')
        ax.set_xticks(x); ax.set_xticklabels(sim_stats.index, rotation=45)
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1]
        colors = plt.cm.Set3(range(len(sim_stats)))
        wedges, texts, autotexts = ax.pie(
            sim_stats['n_frames'], labels=sim_stats.index, autopct='%1.1f%%',
            colors=colors, startangle=90, pctdistance=0.85,
        )
        ax.set_title('Frame Count Percentage Distribution')
        for autotext in autotexts:
            autotext.set_fontsize(9); autotext.set_fontweight('bold')
        for wedge, count in zip(wedges, sim_stats['n_frames']):
            ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            y = np.sin(np.deg2rad(ang)); x = np.cos(np.deg2rad(ang))
            ha = {-1: "right", 1: "left"}[int(np.sign(x))]
            ax.annotate(f'{count:,}', xy=(x, y),
                        xytext=(1.25 * np.sign(x), 1.35 * y),
                        horizontalalignment=ha, fontsize=8, va='center')
        plt.tight_layout()
        plt.savefig(fig_dir / '01_configuration_count.png', bbox_inches='tight')
        print(f"Saved: {fig_dir / '01_configuration_count.png'}")
        plt.close()

    def plot_overall_distributions(self, fig_dir):
        if 'force_mean' not in self.summary_df.columns:
            print("No force data available for plotting"); return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Overall Force and Energy Distributions',
                     fontsize=14, fontweight='bold')

        ax = axes[0]
        weights = self.summary_df['n_particles']
        force_data = self.summary_df['force_mean']
        f_min, f_max = force_data.min(), force_data.max()
        f_margin = (f_max - f_min) * 0.1
        ax.hist(force_data, bins=60, weights=weights,
                alpha=0.6, edgecolor='black', label='Histogram')
        force_x = np.linspace(f_min - f_margin, f_max + f_margin, 200)
        sample_forces = np.repeat(force_data.values, weights.astype(int))
        if len(sample_forces) > 10000:
            sample_forces = np.random.choice(sample_forces, 10000)
        try:
            kde = gaussian_kde(sample_forces)
            ax2 = ax.twinx()
            ax2.plot(force_x, kde(force_x), 'r-', linewidth=2, label='KDE')
            ax2.set_ylabel('Density', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')
            ax.set_xlim(f_min - f_margin, f_max + f_margin)
        except Exception:
            pass
        ax.set_xlabel('Mean Force'); ax.set_ylabel('Total Particles')
        ax.set_title('Force Distribution (All Data)')
        ax.axvline(force_data.mean(), color='green', linestyle='--',
                   label=f'Mean: {force_data.mean():.4f}', alpha=0.7)
        ax.legend(loc='upper left'); ax.grid(alpha=0.3)

        ax = axes[1]
        if 'pe_mean' in self.summary_df.columns:
            pe_data = self.summary_df['pe_mean']
            pe_min, pe_max = pe_data.min(), pe_data.max()
            pe_margin = (pe_max - pe_min) * 0.1
            pe_x = np.linspace(pe_min - pe_margin, pe_max + pe_margin, 200)
            ax.hist(pe_data, bins=60, weights=weights,
                    alpha=0.6, edgecolor='black', label='Histogram')
            sample_pes = np.repeat(pe_data.values, weights.astype(int))
            if len(sample_pes) > 10000:
                sample_pes = np.random.choice(sample_pes, 10000)
            try:
                kde = gaussian_kde(sample_pes)
                ax2 = ax.twinx()
                ax2.plot(pe_x, kde(pe_x), 'r-', linewidth=2, label='KDE')
                ax2.set_ylabel('Density', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.legend(loc='upper right')
                ax.set_xlim(pe_min - pe_margin, pe_max + pe_margin)
            except Exception:
                pass
            ax.set_xlabel('Mean Potential Energy'); ax.set_ylabel('Total Particles')
            ax.set_title('Energy Distribution (All Data)')
            ax.axvline(pe_data.mean(), color='green', linestyle='--',
                       label=f'Mean: {pe_data.mean():.4f}', alpha=0.7)
            ax.legend(loc='upper left'); ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No energy data', ha='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(fig_dir / '02_overall_distributions.png', bbox_inches='tight')
        print(f"Saved: {fig_dir / '02_overall_distributions.png'}")
        plt.close()

    def plot_by_sim_type(self, fig_dir):
        if 'force_mean' not in self.summary_df.columns:
            print("No force data available for plotting"); return
        sim_types = sorted(self.summary_df['sim_type'].unique())
        fig, axes = plt.subplots(len(sim_types), 2, figsize=(14, 4 * len(sim_types)))
        if len(sim_types) == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('Force and Energy Distributions by Simulation Type',
                     fontsize=14, fontweight='bold')
        for i, sim_type in enumerate(sim_types):
            data = self.summary_df[self.summary_df['sim_type'] == sim_type]
            weights = data['n_particles']
            force_data = data['force_mean']

            ax = axes[i, 0]
            ax.hist(force_data, bins=40, weights=weights, alpha=0.6,
                    edgecolor='black', color=f'C{i}')
            ax.set_xlabel('Mean Force'); ax.set_ylabel('Total Particles')
            ax.set_title(f'{sim_type} - Force Distribution')
            ax.axvline(force_data.mean(), color='red', linestyle='--',
                       label=f'Mean: {force_data.mean():.4f}')
            ax.legend(); ax.grid(alpha=0.3)

            ax = axes[i, 1]
            if 'pe_mean' in data.columns:
                pe_data = data['pe_mean']
                ax.hist(pe_data, bins=40, weights=weights, alpha=0.6,
                        edgecolor='black', color=f'C{i}')
                ax.set_xlabel('Mean Potential Energy'); ax.set_ylabel('Total Particles')
                ax.set_title(f'{sim_type} - Energy Distribution')
                ax.axvline(pe_data.mean(), color='red', linestyle='--',
                           label=f'Mean: {pe_data.mean():.4f}')
                ax.legend(); ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No energy data', ha='center', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(fig_dir / '03_by_sim_type.png', bbox_inches='tight')
        print(f"Saved: {fig_dir / '03_by_sim_type.png'}")
        plt.close()

    def plot_temperature_dependence(self, fig_dir):
        if 'force_mean' not in self.summary_df.columns:
            print("No force data available for plotting"); return
        temp_data = self.summary_df[self.summary_df['temperature'].notna()].copy()
        if len(temp_data) == 0:
            print("No temperature data available for plotting"); return
        temp_sims = sorted([s for s in temp_data['sim_type'].unique()
                            if 'npt' in s or 'nvt' in s])
        if len(temp_sims) == 0:
            print("No NPT/NVT simulations with temperature data found"); return

        fig, axes = plt.subplots(len(temp_sims), 2, figsize=(14, 5 * len(temp_sims)))
        if len(temp_sims) == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('Temperature Dependence of Force and Energy Distributions',
                     fontsize=14, fontweight='bold')
        for i, sim_type in enumerate(temp_sims):
            sim_df = temp_data[temp_data['sim_type'] == sim_type].sort_values('temperature')
            temps = sim_df['temperature'].unique()
            ax = axes[i, 0]
            force_data_by_temp = [sim_df[sim_df['temperature'] == t]['force_mean'].values
                                  for t in sorted(temps)]
            bp = ax.boxplot(force_data_by_temp,
                            labels=[f'{int(t)}' for t in sorted(temps)],
                            patch_artist=True, showmeans=True)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(temps)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Mean Force')
            ax.set_title(f'{sim_type} - Force Distribution by Temperature')
            ax.grid(alpha=0.3, axis='y')

            ax = axes[i, 1]
            if 'pe_mean' in sim_df.columns:
                pe_data_by_temp = [sim_df[sim_df['temperature'] == t]['pe_mean'].values
                                   for t in sorted(temps)]
                bp = ax.boxplot(pe_data_by_temp,
                                labels=[f'{int(t)}' for t in sorted(temps)],
                                patch_artist=True, showmeans=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color); patch.set_alpha(0.7)
                ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Mean Potential Energy')
                ax.set_title(f'{sim_type} - Energy Distribution by Temperature')
                ax.grid(alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No energy data', ha='center', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(fig_dir / '04_temperature_dependence.png', bbox_inches='tight')
        print(f"Saved: {fig_dir / '04_temperature_dependence.png'}")
        plt.close()

    # ------------------------------------------------------------------
    def generate_summary_report(self):
        report_path = self.output_dir / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("COARSE-GRAINED DATA ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write("DATASET OVERVIEW\n" + "-" * 40 + "\n")
            f.write(f"Total files processed: {len(self.summary_df)}\n")
            f.write(f"Total particles:       {self.summary_df['n_particles'].sum():,}\n")
            f.write(f"Total frames:          {self.summary_df['n_frames'].sum():,}\n")
            f.write(f"Average particles per frame: {self.summary_df['n_particles'].mean():.1f}\n\n")

            f.write("BY SIMULATION TYPE\n" + "-" * 40 + "\n")
            for sim_type in sorted(self.summary_df['sim_type'].unique()):
                data = self.summary_df[self.summary_df['sim_type'] == sim_type]
                f.write(f"\n{sim_type}:\n  Files: {len(data)}\n")
                f.write(f"  Particles: {data['n_particles'].sum():,}\n")
                f.write(f"  Frames: {data['n_frames'].sum():,}\n")
                pct = data['n_particles'].sum() / self.summary_df['n_particles'].sum() * 100
                f.write(f"  Percentage: {pct:.1f}%\n")

            temp_data = self.summary_df[self.summary_df['temperature'].notna()]
            if len(temp_data) > 0:
                f.write("\n\nBY TEMPERATURE\n" + "-" * 40 + "\n")
                temp_summary = temp_data.groupby('temperature').agg({
                    'n_particles': 'sum', 'n_frames': 'sum',
                }).sort_index()
                for temp, row in temp_summary.iterrows():
                    f.write(f"\n{int(temp)} K:\n  Particles: {row['n_particles']:,}\n"
                            f"  Frames: {row['n_frames']:,}\n")

            if 'force_mean' in self.summary_df.columns:
                f.write("\n\nFORCE STATISTICS\n" + "-" * 40 + "\n")
                weights = self.summary_df['n_particles']
                f_mean = np.average(self.summary_df['force_mean'], weights=weights)
                f_std = np.average(self.summary_df['force_std'], weights=weights)
                f.write(f"Weighted mean force: {f_mean:.6f}\n")
                f.write(f"Weighted std force:  {f_std:.6f}\n")
                f.write(f"Global min force: {self.summary_df['force_min'].min():.6f}\n")
                f.write(f"Global max force: {self.summary_df['force_max'].max():.6f}\n")

            if 'pe_mean' in self.summary_df.columns:
                f.write("\n\nENERGY STATISTICS\n" + "-" * 40 + "\n")
                weights = self.summary_df['n_particles']
                pe_mean = np.average(self.summary_df['pe_mean'], weights=weights)
                pe_std = np.average(self.summary_df['pe_std'], weights=weights)
                f.write(f"Weighted mean PE: {pe_mean:.6f}\n")
                f.write(f"Weighted std PE:  {pe_std:.6f}\n")
                f.write(f"Global min PE: {self.summary_df['pe_min'].min():.6f}\n")
                f.write(f"Global max PE: {self.summary_df['pe_max'].max():.6f}\n")
            f.write("\n" + "=" * 70 + "\n")
        print(f"Saved: {report_path}")

    def save_summary_csv(self):
        csv_path = self.output_dir / 'analysis_summary.csv'
        self.summary_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    def run_full_analysis(self, sample_frac=None, max_files=None):
        print("=" * 60)
        print("COARSE-GRAINED DATA ANALYSIS")
        print("=" * 60)
        print(f"Input directory:  {self.base_dir}")
        print(f"Output directory: {self.output_dir}\n")

        self.analyze_dataset(sample_frac=sample_frac, max_files=max_files)
        if len(self.summary_df) == 0:
            print("No data to analyze!"); return

        print("\nGenerating plots and reports...")
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        self.plot_configuration_count(fig_dir)
        self.plot_overall_distributions(fig_dir)
        self.plot_by_sim_type(fig_dir)
        self.plot_temperature_dependence(fig_dir)
        self.generate_summary_report()
        self.save_summary_csv()
        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)


def json_load(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# Entry point
# =============================================================================

def run(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """cgkit analyze-cg entry point."""
    _import_deps()  # populate plt / gridspec / gaussian_kde

    paths = config.get('paths', {})
    analysis_cfg = config.get('analysis_cg', {}) or {}

    base_dir = paths.get('cg_data_base_dir', '/mnt/d/Workbench/CH_CG/02.cg_dataset')
    output_dir = analysis_cfg.get('output_dir') or paths.get('analysis_output_base_dir')
    sample = getattr(args, 'sample', None) or analysis_cfg.get('sample')
    max_files = getattr(args, 'max_files', None) or analysis_cfg.get('max_files')

    analyzer = CGDataAnalyzer(base_dir=base_dir, output_dir=output_dir, config=config)
    analyzer.run_full_analysis(sample_frac=sample, max_files=max_files)
    return 0
