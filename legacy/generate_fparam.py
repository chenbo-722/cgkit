#!/usr/bin/env python3
"""
Generate fparam.raw / fparam.npy for DeepMD-kit training data.

Two modes (subcommands):
  extract : Extract per-frame temperature from LAMMPS log.lammps files
            (for varying-T simulations like 3-upT / 4-dnT).
  const   : Fill fparam with a constant temperature for each {sim}/{temp}
            directory (for NPT / NVT simulations).

Temperature unit can be selected with --unit:
  K  : Kelvin (raw LAMMPS thermo output)
  eV : kT = kB * T  (Boltzmann constant in eV/K)

Both raw text and numpy (.npy) outputs are written.
"""
import os
import argparse
import numpy as np

# Boltzmann constant in eV/K (CODATA)
K_B_eV_per_K = 8.617333262e-5


def parse_lammps_log(log_file):
    """
    Parse a LAMMPS log file and extract the per-step temperature column.

    Returns a list of float temperatures (in Kelvin) in the order they appear.
    """
    temps = []
    in_thermo = False
    temp_idx = None

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect thermo header (must contain both Step and Temp)
            if not in_thermo and 'Step' in line and 'Temp' in line:
                headers = line.split()
                if 'Temp' in headers:
                    temp_idx = headers.index('Temp')
                    in_thermo = True
                    continue

            if in_thermo:
                parts = line.split()
                if parts and parts[0].lstrip('-').isdigit():
                    try:
                        temps.append(float(parts[temp_idx]))
                    except (ValueError, IndexError):
                        pass
                elif line.startswith('Loop time') or not parts:
                    # End of this thermo block
                    in_thermo = False

    return temps


def to_output_unit(values_kelvin, unit):
    """Convert an array-like of Kelvin temperatures to the requested unit."""
    arr = np.asarray(values_kelvin, dtype=np.float64)
    if unit == 'eV':
        return arr * K_B_eV_per_K
    return arr


def write_fparam(values, raw_path, npy_path):
    """Write fparam.raw (one value per line) and fparam.npy (1D float64)."""
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)

    with open(raw_path, 'w') as f:
        for v in values:
            f.write(f"{v:.10g}\n")

    np.save(npy_path, np.asarray(values, dtype=np.float64))


def run_extract(args):
    """Extract T(t) from LAMMPS log files."""
    total = 0
    for sim_name in args.sim:
        log_file = os.path.join(args.log_dir, sim_name, 'log.lammps')
        raw_file = os.path.join(args.output_dir, sim_name, 'fparam.raw')
        npy_file = os.path.join(args.output_dir, sim_name, 'set.000', 'fparam.npy')

        if not os.path.exists(log_file):
            print(f"[skip] Log not found: {log_file}")
            continue

        print(f"[extract] {sim_name}  <- {log_file}")
        temps_K = parse_lammps_log(log_file)
        if not temps_K:
            print(f"  Warning: no temperature values found in {log_file}")
            continue

        values = to_output_unit(temps_K, args.unit)
        write_fparam(values, raw_file, npy_file)

        print(f"  frames : {len(values)}")
        print(f"  range  : {values.min():.6g} - {values.max():.6g} {args.unit}")
        print(f"  -> {raw_file}")
        print(f"  -> {npy_file}")
        total += len(values)

    print(f"\nTotal frames written (extract): {total}")


def run_const(args):
    """Generate constant-T fparam matched to box.raw frame count."""
    for sim_name in args.sim:
        print(f"[const] {sim_name}")
        for temperature in args.temp:
            temp_dir = os.path.join(args.base_dir, sim_name, str(temperature))
            box_file = os.path.join(temp_dir, 'box.raw')
            raw_file = os.path.join(temp_dir, 'fparam.raw')
            npy_file = os.path.join(temp_dir, 'set.000', 'fparam.npy')

            if not os.path.exists(temp_dir):
                print(f"  [skip] Directory not found: {temp_dir}")
                continue
            if not os.path.exists(box_file):
                print(f"  [skip] box.raw not found: {box_file}")
                continue

            with open(box_file, 'r') as f:
                n_frames = sum(1 for line in f if line.strip())

            value_K = float(temperature)
            value = to_output_unit([value_K], args.unit)[0]
            values = np.full(n_frames, value, dtype=np.float64)
            write_fparam(values, raw_file, npy_file)

            print(f"  T={value_K} K  ->  {value:.10g} {args.unit}  "
                  f"({n_frames} frames)")
            print(f"    -> {raw_file}")
            print(f"    -> {npy_file}")


def add_unit_arg(p):
    p.add_argument(
        '--unit', choices=['K', 'eV'], default='eV',
        help='Output unit for fparam. K = Kelvin; eV = kT = kB*T '
             '(default: eV)'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Generate fparam (frame parameters) for DeepMD-kit training data.'
    )
    subparsers = parser.add_subparsers(
        dest='mode', required=True,
        help='Generation mode'
    )

    # ---- extract mode ----
    p_ex = subparsers.add_parser(
        'extract',
        help='Extract per-frame T from LAMMPS log.lammps (varying-T sims)'
    )
    p_ex.add_argument('--log-dir', default='/mnt/d/Workbench/CH_CG/01.aa',
                      help='Base directory containing <sim>/log.lammps')
    p_ex.add_argument('--output-dir',
                      default='/mnt/d/Workbench/CH_CG/03.cg_npy/training_data',
                      help='Output training_data directory')
    p_ex.add_argument('--sim', nargs='+', default=['3-upT', '4-dnT'],
                      help='Simulation subdirectory names under --log-dir')
    add_unit_arg(p_ex)
    p_ex.set_defaults(func=run_extract)

    # ---- const mode ----
    p_c = subparsers.add_parser(
        'const',
        help='Fill fparam with a constant T per {sim}/{temp} directory'
    )
    p_c.add_argument('--base-dir',
                     default='/mnt/d/Workbench/CH_CG/03.cg_npy/training_data',
                     help='Base training_data directory')
    p_c.add_argument('--sim', nargs='+', default=['1-npt', '2-nvt'],
                     help='Simulation names to process')
    p_c.add_argument('--temp', type=float, nargs='+',
                     default=[200, 300, 400, 500, 600],
                     help='Temperatures in Kelvin (one directory each)')
    add_unit_arg(p_c)
    p_c.set_defaults(func=run_const)

    args = parser.parse_args()
    if args.unit == 'eV':
        print(f"Unit: eV  (kB = {K_B_eV_per_K} eV/K, value = kB*T)")
    else:
        print("Unit: K  (Kelvin)")
    args.func(args)
    print("\nDone!")


if __name__ == '__main__':
    main()
