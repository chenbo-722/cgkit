#!/usr/bin/env python3
"""cgkit - single-entry dispatcher for the CH_CG cgkit suite.

Subcommands (each maps 1:1 to a former standalone script):

    cgkit cg-gen          # legacy 02-get_CGdata_parall.py
    cgkit to-deepmd       # legacy 03-trans_CGnpy_parall.py
    cgkit fparam extract  # legacy generate_fparam.py extract
    cgkit fparam const    # legacy generate_fparam.py const
    cgkit analyze-cg      # legacy 0x-analyze_cg_data.py
    cgkit analyze-atomic  # legacy 0x-analyze_atomic_structure.py
    cgkit plot-pt         # new: P/T coverage scatter from log.lammps

Design: top-level imports are limited to numpy/pandas-grade modules so that
``cgkit --help`` and all non-analysis subcommands work without matplotlib /
scipy / sklearn / networkx / ase / dscribe / torch. ``analyze-atomic`` is
loaded via a lazy trampoline so its heavy deps are only imported on demand.
"""
from __future__ import annotations

import sys
from typing import Optional

# Top-level: CLI + domain modules that need only numpy/pandas/tqdm.
from cglib.cli import build_parser
from cglib.config import load_config, merge_config_with_args, default_config_path
from cglib import cg_gen, deepmd_conv, fparam, analyze_cg, pt_plot


# =============================================================================
# Lazy trampoline for analyze-atomic (heavy deps)
# =============================================================================

def _run_analyze_atomic(config, args) -> int:
    """Import cglib.analyze_atomic on first call to keep top-level imports cheap."""
    from cglib import analyze_atomic
    return analyze_atomic.run(config, args)


# =============================================================================
# Dispatch table
# =============================================================================

# Each entry: (handler_callable, requires_fparam_mode).
DISPATCH = {
    "cg-gen":          (cg_gen.run,           False),
    "to-deepmd":       (deepmd_conv.run,      False),
    "fparam":          (fparam.run,           True),
    "analyze-cg":      (analyze_cg.run,       False),
    "analyze-atomic":  (_run_analyze_atomic,  False),
    "plot-pt":         (pt_plot.run,          False),
}


# =============================================================================
# Entry point
# =============================================================================

def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command
    if command not in DISPATCH:
        parser.error(f"Unknown command: {command!r}")

    # Load + merge config.
    config_path = getattr(args, "config", None) or default_config_path()
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        parser.error(f"Config file not found: {config_path}")
    config = merge_config_with_args(config, args)

    # Dispatch.
    handler, _ = DISPATCH[command]
    return int(handler(config, args) or 0)


if __name__ == "__main__":
    sys.exit(main())
