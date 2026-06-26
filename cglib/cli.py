"""argparse builder for cgkit.

Single ``build_parser()`` returns the top-level ``ArgumentParser`` with one
subparser per cgkit subcommand. Common args (``--config``, ``--sim``, ``--temp``,
``--workers``) are added to every subparser via :func:`add_common_args`.
Subcommand-specific args (``--base-dir``, ``--output-dir``, ``--mode`` ...)
live on their own subparsers.
"""
from __future__ import annotations

import argparse
from typing import Iterable


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add args shared by every cgkit subcommand."""
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to config.json (default: cgkit/config.json)')
    parser.add_argument('--sim', type=str, nargs='+', default=None, metavar='NAME',
                        help='Process only the named simulation(s), e.g. '
                             '--sim 1-npt or --sim 3-upT 4-dnT.')
    parser.add_argument('--temp', type=int, nargs='+', default=None, metavar='K',
                        help='Override temperatures (e.g. --temp 200 300 400).')
    parser.add_argument('--workers', '-w', type=int, default=None, metavar='N',
                        help='Number of parallel worker processes '
                             '(default: auto-detect).')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='cgkit',
        description='CH_CG coarse-graining toolkit (single entry for the former '
                    '02-/03-/0x-/generate_fparam scripts).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  cgkit cg-gen                              # full CG generation from 01.aa
  cgkit cg-gen --sim 1-npt --temp 200 300   # only NPT @ 200K/300K
  cgkit to-deepmd                           # CG CSV -> DeepMD .raw/.npy
  cgkit fparam extract                      # T(t) from LAMMPS logs -> fparam
  cgkit fparam const --unit K               # constant-T fparam in Kelvin
  cgkit analyze-cg                          # statistical CG analysis
  cgkit analyze-atomic --mode cg            # SOAP/PCA/t-SNE on CG trajectories
  cgkit plot-pt                             # P-T coverage scatter from log.lammps
""",
    )
    sub = parser.add_subparsers(dest='command', required=True, metavar='<command>')

    # --- cg-gen ------------------------------------------------------------
    p_cggen = sub.add_parser(
        'cg-gen',
        help='LAMMPS trajectories -> coarse-grained CSV (legacy 02-)',
        description='Read LAMMPS dumps from paths.base_dir and write CG particle/'
                    'box CSV files (legacy 02-get_CGdata_parall).',
    )
    add_common_args(p_cggen)
    p_cggen.add_argument('--base-dir', type=str, default=None,
                         help='Override paths.base_dir (01.aa atomic simulations).')
    p_cggen.add_argument('--output-dir', type=str, default=None,
                         help='Override paths.cg_data_base_dir (CG CSV output).')

    # --- to-deepmd ---------------------------------------------------------
    p_deepmd = sub.add_parser(
        'to-deepmd',
        help='CG CSV -> DeepMD-kit .raw/.npy (legacy 03-)',
        description='Convert CG particle/box CSV files into DeepMD-kit training '
                    'data (.raw + set.NNN/*.npy).',
    )
    add_common_args(p_deepmd)
    p_deepmd.add_argument('--base-dir', type=str, default=None,
                          help='Override paths.cg_data_base_dir (input CG CSV).')
    p_deepmd.add_argument('--output-dir', type=str, default=None,
                          help='Override paths.deepmd_output_base_dir (output).')

    # --- fparam (extract / const) ------------------------------------------
    p_fparam = sub.add_parser(
        'fparam',
        help='Generate DeepMD fparam.raw/.npy (legacy generate_fparam)',
        description='Generate DeepMD-kit fparam files. Two modes: '
                    '"extract" reads T(t) from LAMMPS logs, "const" writes '
                    'constant-T frames matched to box.raw.',
    )
    fparam_sub = p_fparam.add_subparsers(
        dest='fparam_mode', required=True, metavar='<mode>',
    )

    p_ext = fparam_sub.add_parser(
        'extract', help='Extract per-frame T(t) from log.lammps.')
    add_common_args(p_ext)
    p_ext.add_argument('--unit', choices=['K', 'eV'], default=None,
                       help='Output unit (default: from config, typically eV).')
    p_ext.add_argument('--log-dir', type=str, default=None,
                       help='Override paths.log_dir (where log.lammps lives).')
    p_ext.add_argument('--output-dir', type=str, default=None,
                       help='Override paths.deepmd_output_base_dir.')

    p_const = fparam_sub.add_parser(
        'const', help='Generate constant-T fparam matched to box.raw.')
    add_common_args(p_const)
    p_const.add_argument('--unit', choices=['K', 'eV'], default=None,
                         help='Output unit (default: from config, typically eV).')
    p_const.add_argument('--base-dir', type=str, default=None,
                         help='Override paths.deepmd_output_base_dir.')

    # --- analyze-cg --------------------------------------------------------
    p_anacg = sub.add_parser(
        'analyze-cg',
        help='Statistical analysis of CG CSV data (legacy 0x-analyze_cg_data)',
        description='Compute RDF/energy/temperature statistics from CG CSV '
                    'particle files and render overview plots.',
    )
    add_common_args(p_anacg)
    p_anacg.add_argument('--base-dir', type=str, default=None,
                         help='Override paths.cg_data_base_dir (input).')
    p_anacg.add_argument('--output-dir', type=str, default=None,
                         help='Override analysis_cg.output_dir.')

    # --- analyze-atomic ----------------------------------------------------
    p_anat = sub.add_parser(
        'analyze-atomic',
        help='SOAP/PCA/t-SNE/GNN analysis (legacy 0x-analyze_atomic_structure)',
        description='Run SOAP descriptor + PCA + t-SNE + clustering pipeline on '
                    'either CG trajectories (mode=cg) or atomic dumps (mode=aa).',
    )
    add_common_args(p_anat)
    p_anat.add_argument('--mode', choices=['cg', 'aa'], default=None,
                        help='Analysis mode (default: from config).')
    p_anat.add_argument('--base-dir', type=str, default=None,
                        help='Override CG/AA base dir (chosen by --mode).')
    p_anat.add_argument('--output-dir', type=str, default=None,
                        help='Override analysis_atomic.output_dir.')
    p_anat.add_argument('--max-frames', type=int, default=None, metavar='N',
                        help='Cap on total frames analysed.')
    p_anat.add_argument('--max-per-file', type=int, default=None, metavar='N',
                        help='Cap on frames per trajectory file (CG mode only).')

    # --- plot-pt -----------------------------------------------------------
    p_pt = sub.add_parser(
        'plot-pt',
        help='P/T overview from log.lammps (new module)',
        description='Join every AA dump frame to its thermo row in '
                    'log.lammps and render a single P-vs-T scatter. Useful '
                    'for spotting holes in the (P, T) coverage of the '
                    'training set before fitting a CG potential.',
    )
    add_common_args(p_pt)
    p_pt.add_argument('--base-dir', type=str, default=None,
                      help='Override paths.aa_data_base_dir (AA dump root).')
    p_pt.add_argument('--output-dir', type=str, default=None,
                      help='Override plot_pt.output_dir (CSV + PNG destination).')
    p_pt.add_argument('--log-dir', type=str, default=None,
                      help='Override paths.log_dir (root holding <sim>/log.lammps).')
    p_pt.add_argument('--max-frames', type=int, default=None, metavar='N',
                      help='Cap on total frames plotted (uniform downsample).')

    return parser


def parse_argv(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Convenience: build parser + parse_args in one call."""
    return build_parser().parse_args(argv)
