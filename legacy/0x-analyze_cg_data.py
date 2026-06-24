#!/usr/bin/env python3
"""
Coarse-Grained Data Analysis Script

Analyzes force and energy distributions in CG dataset, and summarizes
data composition by simulation type and temperature.

Author: CH_CG Workflow
Date: 2025-12-31
"""

import os
import json
import argparse
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import gaussian_kde

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False


class CGDataAnalyzer:
    """Analyzer for coarse-grained particle data"""

    def __init__(self, base_dir, output_dir=None, config_file=None):
        """
        Initialize the analyzer.

        Args:
            base_dir: Base directory containing CG data (default: 02.cg_dataset)
            output_dir: Directory to save analysis results
            config_file: Optional config.json file for simulation info
        """
        self.base_dir = Path(base_dir)

        # Load config for simulation metadata and output directory
        self.config = None
        config_output_dir = None

        # Try to load from provided config file
        if config_file:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    if 'paths' in self.config and 'output_base_dir' in self.config['paths']:
                        config_output_dir = self.config['paths']['output_base_dir']

        # Try default config location
        if self.config is None:
            default_config = Path(__file__).parent / "config.json"
            if default_config.exists():
                with open(default_config, 'r') as f:
                    self.config = json.load(f)
                    if 'paths' in self.config and 'output_base_dir' in self.config['paths']:
                        config_output_dir = self.config['paths']['output_base_dir']

        # Set output directory (priority: CLI arg > config > default)
        if output_dir:
            self.output_dir = Path(output_dir)
        elif config_output_dir:
            self.output_dir = Path(config_output_dir)
        else:
            self.output_dir = self.base_dir / "analysis_results"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.all_data = {}
        self.summary_stats = {}

    def find_particle_files(self):
        """Find all particle CSV files in the dataset."""
        # Use recursive glob to handle different directory structures
        pattern = str(self.base_dir / "**/*_particles.csv")
        files = glob.glob(pattern, recursive=True)
        return sorted(files)

    def parse_file_info(self, filepath):
        """
        Parse simulation type and temperature from filepath.

        Expected formats:
        - base_dir/sim_type/temp/filename_particles.csv (1-npt, 2-nvt)
        - base_dir/sim_type/filename_particles.csv (3-upT, 4-dnT)

        Returns:
            dict with keys: sim_type, temperature, filename
        """
        path = Path(filepath)
        parts = path.relative_to(self.base_dir).parts

        info = {
            'filepath': filepath,
            'filename': path.stem,
            'sim_type': parts[0] if len(parts) > 0 else 'unknown',
            'temperature': None,
        }

        # Try to extract temperature from path (second level for 1-npt, 2-nvt)
        if len(parts) > 1:
            try:
                temp_str = parts[1]
                if temp_str.isdigit():
                    info['temperature'] = int(temp_str)
            except ValueError:
                pass

        # Try to extract from filename pattern (e.g., NPT.200.xxx, upT.100000)
        filename = path.stem
        for part in filename.split('.'):
            if part.isdigit():
                temp = int(part)
                if 100 <= temp <= 1000:  # Reasonable temperature range
                    info['temperature'] = temp
                    break

        return info

    def load_file(self, filepath):
        """Load a single particle CSV file."""
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            return None

    def process_single_file(self, filepath, sample_frac=None):
        """
        Process a single CSV file and extract statistics.

        Args:
            filepath: Path to particle CSV file
            sample_frac: Fraction of data to sample (for large datasets)

        Returns:
            dict with statistics for this file
        """
        df = self.load_file(filepath)
        if df is None or len(df) == 0:
            return None

        # Optional sampling for memory efficiency
        if sample_frac is not None and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)

        info = self.parse_file_info(filepath)

        # Basic info
        stats = {
            'sim_type': info['sim_type'],
            'temperature': info['temperature'],
            'filename': info['filename'],
            'n_particles': len(df),
            'n_frames': df['timestep'].nunique() if 'timestep' in df.columns else 1,
        }

        # Force statistics
        if 'fx' in df.columns:
            f_mag = np.sqrt(df['fx']**2 + df['fy']**2 + df['fz']**2)
            stats.update({
                'force_mean': f_mag.mean(),
                'force_std': f_mag.std(),
                'force_min': f_mag.min(),
                'force_max': f_mag.max(),
                'force_q25': f_mag.quantile(0.25),
                'force_q50': f_mag.quantile(0.50),
                'force_q75': f_mag.quantile(0.75),
                'force_q95': f_mag.quantile(0.95),
                'force_q99': f_mag.quantile(0.99),
            })

        # Energy statistics
        if 'c_pe' in df.columns:
            stats.update({
                'pe_mean': df['c_pe'].mean(),
                'pe_std': df['c_pe'].std(),
                'pe_min': df['c_pe'].min(),
                'pe_max': df['c_pe'].max(),
            })

        return stats

    def analyze_dataset(self, sample_frac=None, max_files=None):
        """
        Analyze the entire dataset.

        Args:
            sample_frac: Fraction of rows to sample per file
            max_files: Maximum number of files to process
        """
        print("Searching for particle data files...")
        files = self.find_particle_files()
        print(f"Found {len(files)} particle files")

        if max_files:
            files = files[:max_files]
            print(f"Processing first {max_files} files")

        all_stats = []
        for i, filepath in enumerate(files):
            if (i + 1) % 50 == 0:
                print(f"Processing file {i+1}/{len(files)}...")

            stats = self.process_single_file(filepath, sample_frac=sample_frac)
            if stats:
                all_stats.append(stats)

        self.summary_df = pd.DataFrame(all_stats)
        print(f"Successfully processed {len(all_stats)} files")
        print(f"Total particles: {self.summary_df['n_particles'].sum():,}")
        print(f"Total frames: {self.summary_df['n_frames'].sum():,}")

        # Print breakdown by sim type
        print("\nDataset breakdown:")
        for sim_type in sorted(self.summary_df['sim_type'].unique()):
            mask = self.summary_df['sim_type'] == sim_type
            data = self.summary_df[mask]
            print(f"  {sim_type}: {len(data)} files, {data['n_particles'].sum():,} particles")

        return self.summary_df

    def plot_configuration_count(self, fig_dir):
        """
        Plot 1: Frame count and percentage for each simulation type.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Dataset Frame Count Analysis', fontsize=14, fontweight='bold')

        # Calculate statistics by sim type
        sim_stats = self.summary_df.groupby('sim_type').agg({
            'n_frames': 'sum',
        })
        sim_stats = sim_stats.sort_values('n_frames', ascending=False)

        # Left plot: Bar chart with frame counts
        ax = axes[0]
        x = np.arange(len(sim_stats))

        bars = ax.bar(x, sim_stats['n_frames'], alpha=0.8, edgecolor='black')

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, sim_stats['n_frames'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Simulation Type')
        ax.set_ylabel('Frame Count')
        ax.set_title('Frame Count by Simulation Type')
        ax.set_xticks(x)
        ax.set_xticklabels(sim_stats.index, rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Right plot: Pie chart with percentage and count labels
        ax = axes[1]
        colors = plt.cm.Set3(range(len(sim_stats)))

        wedges, texts, autotexts = ax.pie(
            sim_stats['n_frames'],
            labels=sim_stats.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            pctdistance=0.85
        )
        ax.set_title('Frame Count Percentage Distribution')

        # Adjust percentage label properties
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')

        # Add count labels below percentage labels
        for i, (wedge, count) in enumerate(zip(wedges, sim_stats['n_frames'])):
            ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            # Position count labels further out than percentage labels
            ax.annotate(f'{count:,}',
                        xy=(x, y),
                        xytext=(1.25*np.sign(x), 1.35*y),
                        horizontalalignment=horizontalalignment,
                        fontsize=8,
                        va='center')

        plt.tight_layout()
        plt.savefig(fig_dir / '01_configuration_count.png', bbox_inches='tight')
        print(f"Saved: {fig_dir / '01_configuration_count.png'}")
        plt.close()

    def plot_overall_distributions(self, fig_dir):
        """
        Plot 2: Overall force and energy distributions (histogram + KDE curve).
        """
        if 'force_mean' not in self.summary_df.columns:
            print("No force data available for plotting")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Overall Force and Energy Distributions', fontsize=14, fontweight='bold')

        # Left plot: Force distribution
        ax = axes[0]
        weights = self.summary_df['n_particles']
        force_data = self.summary_df['force_mean']

        # Get data range for proper axis limits
        f_min, f_max = force_data.min(), force_data.max()
        f_margin = (f_max - f_min) * 0.1  # 10% margin

        # Histogram
        n, bins, patches = ax.hist(force_data, bins=60, weights=weights,
                                     alpha=0.6, edgecolor='black', label='Histogram')

        # KDE curve
        from scipy.stats import gaussian_kde
        force_x = np.linspace(f_min - f_margin, f_max + f_margin, 200)
        # Weighted KDE
        sample_forces = np.repeat(force_data.values, weights.astype(int))
        if len(sample_forces) > 10000:  # Subsample for KDE if too large
            sample_forces = np.random.choice(sample_forces, 10000)
        try:
            kde = gaussian_kde(sample_forces)
            ax2 = ax.twinx()
            ax2.plot(force_x, kde(force_x), 'r-', linewidth=2, label='KDE')
            ax2.set_ylabel('Density', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')
            # Set x-axis limits to match data range with margin
            ax.set_xlim(f_min - f_margin, f_max + f_margin)
        except:
            pass

        ax.set_xlabel('Mean Force')
        ax.set_ylabel('Total Particles')
        ax.set_title('Force Distribution (All Data)')
        ax.axvline(force_data.mean(), color='green', linestyle='--',
                   label=f'Mean: {force_data.mean():.4f}', alpha=0.7)
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)

        # Right plot: Energy distribution
        ax = axes[1]
        if 'pe_mean' in self.summary_df.columns:
            pe_data = self.summary_df['pe_mean']

            # Get data range for proper axis limits
            pe_min, pe_max = pe_data.min(), pe_data.max()
            pe_margin = (pe_max - pe_min) * 0.1  # 10% margin
            pe_x = np.linspace(pe_min - pe_margin, pe_max + pe_margin, 200)

            # Histogram
            n, bins, patches = ax.hist(pe_data, bins=60, weights=weights,
                                        alpha=0.6, edgecolor='black', label='Histogram')

            # KDE curve - use PE data range, not force data range
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
                # Set x-axis limits to match PE data range with margin
                ax.set_xlim(pe_min - pe_margin, pe_max + pe_margin)
            except:
                pass

            ax.set_xlabel('Mean Potential Energy')
            ax.set_ylabel('Total Particles')
            ax.set_title('Energy Distribution (All Data)')
            ax.axvline(pe_data.mean(), color='green', linestyle='--',
                       label=f'Mean: {pe_data.mean():.4f}', alpha=0.7)
            ax.legend(loc='upper left')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No energy data', ha='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(fig_dir / '02_overall_distributions.png', bbox_inches='tight')
        print(f"Saved: {fig_dir / '02_overall_distributions.png'}")
        plt.close()

    def plot_by_sim_type(self, fig_dir):
        """
        Plot 3: Force and energy distributions for each simulation type (4 separate subplots).
        """
        if 'force_mean' not in self.summary_df.columns:
            print("No force data available for plotting")
            return

        sim_types = sorted(self.summary_df['sim_type'].unique())
        fig, axes = plt.subplots(len(sim_types), 2, figsize=(14, 4*len(sim_types)))
        if len(sim_types) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Force and Energy Distributions by Simulation Type',
                     fontsize=14, fontweight='bold')

        for i, sim_type in enumerate(sim_types):
            mask = self.summary_df['sim_type'] == sim_type
            data = self.summary_df[mask]

            # Force distribution
            ax = axes[i, 0]
            weights = data['n_particles']
            force_data = data['force_mean']

            ax.hist(force_data, bins=40, weights=weights, alpha=0.6,
                    edgecolor='black', color=f'C{i}')
            ax.set_xlabel('Mean Force')
            ax.set_ylabel('Total Particles')
            ax.set_title(f'{sim_type} - Force Distribution')
            ax.axvline(force_data.mean(), color='red', linestyle='--',
                       label=f'Mean: {force_data.mean():.4f}')
            ax.legend()
            ax.grid(alpha=0.3)

            # Energy distribution
            ax = axes[i, 1]
            if 'pe_mean' in data.columns:
                pe_data = data['pe_mean']
                ax.hist(pe_data, bins=40, weights=weights, alpha=0.6,
                        edgecolor='black', color=f'C{i}')
                ax.set_xlabel('Mean Potential Energy')
                ax.set_ylabel('Total Particles')
                ax.set_title(f'{sim_type} - Energy Distribution')
                ax.axvline(pe_data.mean(), color='red', linestyle='--',
                           label=f'Mean: {pe_data.mean():.4f}')
                ax.legend()
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No energy data', ha='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(fig_dir / '03_by_sim_type.png', bbox_inches='tight')
        print(f"Saved: {fig_dir / '03_by_sim_type.png'}")
        plt.close()

    def plot_temperature_dependence(self, fig_dir):
        """
        Plot 4: Temperature dependence for simulations with temperature data (1-npt, 2-nvt).
        Uses violin/box plots to show distributions at each temperature.
        """
        if 'force_mean' not in self.summary_df.columns:
            print("No force data available for plotting")
            return

        # Filter data with temperature info
        temp_data = self.summary_df[self.summary_df['temperature'].notna()].copy()
        if len(temp_data) == 0:
            print("No temperature data available for plotting")
            return

        # Get sim types with temperature data
        temp_sims = temp_data['sim_type'].unique()
        temp_sims = sorted([s for s in temp_sims if 'npt' in s or 'nvt' in s])

        if len(temp_sims) == 0:
            print("No NPT/NVT simulations with temperature data found")
            return

        fig, axes = plt.subplots(len(temp_sims), 2, figsize=(14, 5*len(temp_sims)))
        if len(temp_sims) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Temperature Dependence of Force and Energy Distributions',
                     fontsize=14, fontweight='bold')

        for i, sim_type in enumerate(temp_sims):
            sim_mask = temp_data['sim_type'] == sim_type
            sim_df = temp_data[sim_mask]

            # Sort by temperature
            sim_df = sim_df.sort_values('temperature')
            temps = sim_df['temperature'].unique()

            # Prepare data for violin plot (expand by particle count)
            force_by_temp = []
            pe_by_temp = []
            temp_labels = []

            for temp in temps:
                temp_df = sim_df[sim_df['temperature'] == temp]
                # Repeat values based on particle count for proper weighting
                for _, row in temp_df.iterrows():
                    count = int(min(row['n_particles'], 100))  # Cap at 100 for efficiency
                    force_by_temp.extend([row['force_mean']] * count)
                    if 'pe_mean' in row:
                        pe_by_temp.extend([row['pe_mean']] * count)
                temp_labels.extend([f'{int(temp)}K'] * len(temp_df) * min(100, temp_df['n_particles'].min()))

            # Force vs Temperature (box plot)
            ax = axes[i, 0]
            force_data_by_temp = [sim_df[sim_df['temperature'] == t]['force_mean'].values
                                   for t in sorted(temps)]

            bp = ax.boxplot(force_data_by_temp, labels=[f'{int(t)}' for t in sorted(temps)],
                           patch_artist=True, showmeans=True)

            # Color the boxes
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(temps)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel('Mean Force')
            ax.set_title(f'{sim_type} - Force Distribution by Temperature')
            ax.grid(alpha=0.3, axis='y')

            # Energy vs Temperature (box plot)
            ax = axes[i, 1]
            if 'pe_mean' in sim_df.columns:
                pe_data_by_temp = [sim_df[sim_df['temperature'] == t]['pe_mean'].values
                                   for t in sorted(temps)]

                bp = ax.boxplot(pe_data_by_temp, labels=[f'{int(t)}' for t in sorted(temps)],
                               patch_artist=True, showmeans=True)

                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_xlabel('Temperature (K)')
                ax.set_ylabel('Mean Potential Energy')
                ax.set_title(f'{sim_type} - Energy Distribution by Temperature')
                ax.grid(alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No energy data', ha='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(fig_dir / '04_temperature_dependence.png', bbox_inches='tight')
        print(f"Saved: {fig_dir / '04_temperature_dependence.png'}")
        plt.close()

    def generate_summary_report(self):
        """Generate a text summary report."""
        report_path = self.output_dir / 'analysis_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("COARSE-GRAINED DATA ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Overall statistics
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total files processed: {len(self.summary_df)}\n")
            f.write(f"Total particles: {self.summary_df['n_particles'].sum():,}\n")
            f.write(f"Total frames: {self.summary_df['n_frames'].sum():,}\n")
            f.write(f"Average particles per frame: {self.summary_df['n_particles'].mean():.1f}\n\n")

            # By simulation type
            f.write("BY SIMULATION TYPE\n")
            f.write("-" * 40 + "\n")
            for sim_type in sorted(self.summary_df['sim_type'].unique()):
                mask = self.summary_df['sim_type'] == sim_type
                data = self.summary_df[mask]
                f.write(f"\n{sim_type}:\n")
                f.write(f"  Files: {len(data)}\n")
                f.write(f"  Particles: {data['n_particles'].sum():,}\n")
                f.write(f"  Frames: {data['n_frames'].sum():,}\n")
                pct = data['n_particles'].sum() / self.summary_df['n_particles'].sum() * 100
                f.write(f"  Percentage: {pct:.1f}%\n")

            # By temperature
            temp_data = self.summary_df[self.summary_df['temperature'].notna()]
            if len(temp_data) > 0:
                f.write("\n\nBY TEMPERATURE\n")
                f.write("-" * 40 + "\n")
                temp_summary = temp_data.groupby('temperature').agg({
                    'n_particles': 'sum',
                    'n_frames': 'sum',
                }).sort_index()
                for temp, row in temp_summary.iterrows():
                    f.write(f"\n{int(temp)} K:\n")
                    f.write(f"  Particles: {row['n_particles']:,}\n")
                    f.write(f"  Frames: {row['n_frames']:,}\n")

            # Force statistics
            if 'force_mean' in self.summary_df.columns:
                f.write("\n\nFORCE STATISTICS\n")
                f.write("-" * 40 + "\n")
                weights = self.summary_df['n_particles']
                f_mean_weighted = np.average(self.summary_df['force_mean'], weights=weights)
                f_std_weighted = np.average(self.summary_df['force_std'], weights=weights)
                f_min_global = self.summary_df['force_min'].min()
                f_max_global = self.summary_df['force_max'].max()
                f.write(f"Weighted mean force: {f_mean_weighted:.6f}\n")
                f.write(f"Weighted std force: {f_std_weighted:.6f}\n")
                f.write(f"Global min force: {f_min_global:.6f}\n")
                f.write(f"Global max force: {f_max_global:.6f}\n")

            # Energy statistics
            if 'pe_mean' in self.summary_df.columns:
                f.write("\n\nENERGY STATISTICS\n")
                f.write("-" * 40 + "\n")
                weights = self.summary_df['n_particles']
                pe_mean_weighted = np.average(self.summary_df['pe_mean'], weights=weights)
                pe_std_weighted = np.average(self.summary_df['pe_std'], weights=weights)
                pe_min_global = self.summary_df['pe_min'].min()
                pe_max_global = self.summary_df['pe_max'].max()
                f.write(f"Weighted mean PE: {pe_mean_weighted:.6f}\n")
                f.write(f"Weighted std PE: {pe_std_weighted:.6f}\n")
                f.write(f"Global min PE: {pe_min_global:.6f}\n")
                f.write(f"Global max PE: {pe_max_global:.6f}\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"Saved: {report_path}")

    def save_summary_csv(self):
        """Save summary statistics to CSV."""
        csv_path = self.output_dir / 'analysis_summary.csv'
        self.summary_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    def run_full_analysis(self, sample_frac=None, max_files=None):
        """Run complete analysis pipeline."""
        print("=" * 60)
        print("COARSE-GRAINED DATA ANALYSIS")
        print("=" * 60)
        print(f"Input directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print()

        # Analyze dataset
        self.analyze_dataset(sample_frac=sample_frac, max_files=max_files)

        if len(self.summary_df) == 0:
            print("No data to analyze!")
            return

        print("\nGenerating plots and reports...")
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # Generate the 4 required plots
        self.plot_configuration_count(fig_dir)
        self.plot_overall_distributions(fig_dir)
        self.plot_by_sim_type(fig_dir)
        self.plot_temperature_dependence(fig_dir)

        # Generate reports
        self.generate_summary_report()
        self.save_summary_csv()

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze coarse-grained particle data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all data in default directory
  python analyze_cg_data.py

  # Analyze data from specific directory
  python analyze_cg_data.py --base-dir /path/to/cg_dataset

  # Use sampling for large datasets (10% of data)
  python analyze_cg_data.py --sample 0.1

  # Process only first 100 files
  python analyze_cg_data.py --max-files 100

  # Specify output directory
  python analyze_cg_data.py --output-dir ./my_analysis
        """
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='/mnt/d/Workbench/CH_CG/02.cg_dataset',
        help='Base directory containing CG data (default: 02.cg_dataset)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for analysis results (default: from config.json)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.json file'
    )

    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Fraction of data to sample per file (0.0-1.0) for memory efficiency'
    )

    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process'
    )

    args = parser.parse_args()

    analyzer = CGDataAnalyzer(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        config_file=args.config
    )

    analyzer.run_full_analysis(
        sample_frac=args.sample,
        max_files=args.max_files
    )


if __name__ == '__main__':
    main()
