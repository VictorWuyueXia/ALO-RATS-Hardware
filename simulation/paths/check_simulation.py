"""Run frozen DIRECT cases and independently derive acceptance before the live GUI gate."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from alo_rats_hardware.records import write_json
from simulation.simulation.simulation_cases import CASE_NAMES, simulation_case
from simulation.simulation.validate_simulation import validate_run


def assess(output, returncode):
    """Incomplete evidence or a failed process cannot establish acceptance."""
    try:
        result = validate_run(output)
    except (ValueError, KeyError, FileNotFoundError) as error:
        result = {"accepted": False, "incomplete_evidence": str(error)}
    result["process_returncode"] = returncode
    result["accepted"] &= returncode == 0
    return result


def check_suite(output):
    """A terminated worker fails the suite; never shrink cases or substitute a successful trace."""
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen_cases.json", [simulation_case(name).manifest() for name in CASE_NAMES])
    with (output / "negative_checks.log").open("x") as log:
        negative = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                    "simulation/tests/test_robot_executor.py",
                                    "simulation/tests/test_simulation_workflow.py",
                                    "simulation/tests/test_simulation_acceptance.py",
                                    "simulation/tests/test_simulation_entrypoints.py"],
                                   cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if negative.returncode != 0:
        summary = {"negative_checks_passed": False, "URDF_MPPI_TASK_SUITE_COMPLETE": False}
        write_json(output / "suite_result.json", summary)
        return summary
    worker = """
from simulation.simulation.simulation_isolation import enforce_simulation_isolation
violations = enforce_simulation_isolation()
from simulation.simulation.simulation_app import launch
from pathlib import Path
import sys
launch(sys.argv[1], Path(sys.argv[2]), gui=False, isolation_violations=violations)
"""
    results = {}
    names = (*CASE_NAMES, "centered_rectangle_repeat")
    for index, name in enumerate(names):
        case = "centered_rectangle" if name.endswith("_repeat") else name
        with (output / f"{name}.log").open("x") as log:
            process = subprocess.run([sys.executable, "-c", worker, case, str(output / name)],
                                     cwd=root, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                     stdout=log, stderr=subprocess.STDOUT)
        result = assess(output / name, process.returncode)
        results[name] = result
        (output / name).mkdir(exist_ok=True)
        write_json(output / name / "acceptance.json", result)
        print(name, "PASS" if result["accepted"] else "FAIL", flush=True)
        if process.returncode in (-9, 137):
            for pending in names[index + 1:]:
                results[pending] = {"accepted": False, "not_run": "worker killed; further resource pressure avoided"}
            break
    direct_passed = all(result["accepted"] for result in results.values())
    summary = {"direct_cases": results, "direct_passed": direct_passed,
               "negative_checks_passed": True,
               "gui_completion": False, "URDF_MPPI_TASK_SUITE_COMPLETE": False}
    if direct_passed:
        first, repeat = results["centered_rectangle"], results["centered_rectangle_repeat"]
        summary["repeatability"] = all(first[key] == repeat[key] for key in
                                       ("truth_metrics", "requested_actions", "trajectory_ids", "active_prefixes"))
        process = subprocess.run([sys.executable, "-m", "simulation.paths.run_simulation",
                                  "--case", "centered_rectangle", "--output-dir", str(output / "gui")])
        summary["gui_result"] = assess(output / "gui", process.returncode)
        summary["gui_completion"] = summary["gui_result"]["accepted"]
        summary["URDF_MPPI_TASK_SUITE_COMPLETE"] = summary["repeatability"] and summary["gui_completion"]
    write_json(output / "suite_result.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = check_suite(parser.parse_args().output_dir)
    if not result["URDF_MPPI_TASK_SUITE_COMPLETE"]:
        raise SystemExit("Simulation acceptance failed; inspect suite_result.json and per-case logs.")
