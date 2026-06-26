"""Unified configuration loader for cgkit.

Provides:
- ``load_config``        : read JSON config from disk
- ``merge_config_with_args`` : apply CLI overrides on top of config
- ``get_section``        : dotted-key lookup (e.g. "coarse_graining.patterns")

The loader is command-aware: when applying ``--base-dir`` / ``--output-dir``
overrides, it writes to the *correct* ``paths.*`` key depending on which
subcommand is being run (cg-gen vs to-deepmd vs fparam vs analyze-*).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


# Per-subcommand mapping for --base-dir / --output-dir overrides.
# Keys are (command, cli_attr) -> dotted config key.
COMMAND_PATH_OVERRIDES: Dict[str, Dict[str, str]] = {
    "cg-gen": {
        "base_dir":   "paths.base_dir",
        "output_dir": "paths.cg_data_base_dir",   # cg-gen writes to CG dataset dir
    },
    "to-deepmd": {
        "base_dir":   "paths.cg_data_base_dir",
        "output_dir": "paths.deepmd_output_base_dir",
    },
    "fparam-extract": {
        "log_dir":    "paths.log_dir",
        "output_dir": "paths.deepmd_output_base_dir",
    },
    "fparam-const": {
        "base_dir":   "paths.deepmd_output_base_dir",
    },
    "analyze-cg": {
        "base_dir":   "paths.cg_data_base_dir",
        "output_dir": "analysis_cg.output_dir",
    },
    "analyze-atomic": {
        "base_dir":   None,  # depends on --mode, handled in merge function
        "output_dir": "analysis_atomic.output_dir",
    },
    "plot-pt": {
        "base_dir":   "paths.aa_data_base_dir",
        "output_dir": "plot_pt.output_dir",
        "log_dir":    "paths.log_dir",
    },
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load JSON config. Falls back to DEFAULT_CONFIG_PATH (../config.json)."""
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)
    with open(config_path, "r") as f:
        return json.load(f)


def get_section(config: Dict[str, Any], dotted_key: str,
                default: Any = None) -> Any:
    """Dotted lookup: get_section(cfg, "coarse_graining.patterns")."""
    cur: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Dotted assignment: _set_dotted(cfg, "paths.base_dir", "/foo")."""
    parts = dotted_key.split(".")
    cur = config
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def merge_config_with_args(config: Dict[str, Any],
                           args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides on top of the loaded config (returns same dict).

    Recognised args (all optional):
      args.config    -> reload from this path (handled by caller)
      args.sim       -> filter simulations by name (str OR list for fparam)
      args.temp      -> override temperatures (list[int])
      args.workers   -> processing.max_workers
      args.base_dir  -> paths.<command-specific>
      args.output_dir-> paths.<command-specific>
      args.log_dir   -> paths.log_dir (fparam extract)
      args.mode      -> analyze-atomic mode (cg|aa), chooses base_dir target
    """
    command = getattr(args, "command", None)
    fparam_mode = getattr(args, "fparam_mode", None)
    if command == "fparam" and fparam_mode:
        command_key = f"fparam-{fparam_mode}"
    else:
        command_key = command

    # --sim: filter simulations. fparam uses nargs='+' (list), others single str.
    sim_val = getattr(args, "sim", None)
    if sim_val is not None:
        sim_names = sim_val if isinstance(sim_val, list) else [sim_val]
        if "simulations" in config:
            config["simulations"] = [s for s in config["simulations"]
                                     if s.get("name") in sim_names]

    # --temp: override temperatures for all sims (and fparam.const.temperatures).
    temp_val = getattr(args, "temp", None)
    if temp_val is not None:
        for sim in config.get("simulations", []):
            if sim.get("temperatures") is not None:
                sim["temperatures"] = list(temp_val)
        # fparam const also reads temperatures from its own section
        fp = config.get("fparam", {}).get("const", {})
        if fp:
            fp["temperatures"] = list(temp_val)

    # --workers
    workers = getattr(args, "workers", None)
    if workers is not None:
        config.setdefault("processing", {})["max_workers"] = workers

    # --base-dir / --output-dir / --log-dir, mapped per command.
    # analyze-atomic's base_dir is registered as None in COMMAND_PATH_OVERRIDES
    # because the target key depends on --mode; resolve it here BEFORE the
    # None-skip check, otherwise the CLI override is silently dropped.
    overrides = COMMAND_PATH_OVERRIDES.get(command_key, {})
    for attr, dotted_key in overrides.items():
        value = getattr(args, attr, None)
        if value is None:
            continue
        if command_key == "analyze-atomic" and attr == "base_dir":
            mode = getattr(args, "mode", "cg")
            dotted_key = ("paths.aa_data_base_dir" if mode == "aa"
                          else "paths.cg_data_base_dir")
        if dotted_key is None:
            continue
        _set_dotted(config, dotted_key, value)

    # fparam-specific: --unit
    unit = getattr(args, "unit", None)
    if unit is not None and "fparam" in config:
        config["fparam"]["unit"] = unit

    return config


def default_config_path() -> str:
    """Expose the default config path so cgkit can print it in --help."""
    return str(DEFAULT_CONFIG_PATH)
