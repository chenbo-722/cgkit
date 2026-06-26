"""cglib — shared library for the CH_CG coarse-graining toolkit.

Keep this module empty of imports to ensure ``import cglib`` is cheap and
never pulls heavy optional dependencies (matplotlib / scipy / sklearn /
dscribe / ase / networkx / torch). Each sub-module is imported explicitly
by ``cgkit.py`` on demand.
"""

__all__ = [
    "config",
    "cli",
    "parallel",
    "paths",
    "lammps",
    "io_utils",
    "cg_gen",
    "deepmd_conv",
    "fparam",
    "analyze_cg",
    "analyze_atomic",
    "pt_plot",
]
