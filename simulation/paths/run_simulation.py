"""Approve a nominal scan-defined task and run the live MPPI/URDF simulation."""

import argparse
from pathlib import Path

from simulation.simulation.simulation_isolation import enforce_simulation_isolation


def main():
    violations = enforce_simulation_isolation()
    from simulation.simulation.simulation_app import launch
    from simulation.simulation.simulation_cases import CASE_NAMES
    from simulation.simulation.validate_simulation import validate_run
    from alo_rats_hardware.records import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES, required=True)
    parser.add_argument("--compute", help="Configured compute profile, such as cpu or 1gpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    try:
        launch(args.case, args.output_dir, gui=True, isolation_violations=violations,
               compute_profile=args.compute)
    finally:
        if args.output_dir.is_dir():
            try:
                result = validate_run(args.output_dir)
            except (ValueError, KeyError, FileNotFoundError) as error:
                result = {"accepted": False, "incomplete_evidence": str(error)}
            write_json(args.output_dir / "acceptance.json", result)
    if not result["accepted"]:
        raise SystemExit("Simulation acceptance failed; inspect acceptance.json")


if __name__ == "__main__":
    main()
