"""Run the frozen Phase-0 Boltz inputs singly and as one batch."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run(command: list[str], label: str, log_dir: Path, output_dir: Path, expected_count: int) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    (log_dir / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_dir / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    affinity_jsons = list(output_dir.rglob("affinity_*.json"))
    return {
        "label": label,
        "elapsed_seconds": elapsed,
        "returncode": completed.returncode if len(affinity_jsons) == expected_count else 2,
        "affinity_json_count": len(affinity_jsons),
        "command": command,
    }


def run_with_retries(command: list[str], label: str, log_dir: Path, output_dir: Path, expected_count: int) -> dict:
    attempts = []
    for attempt in range(1, 4):
        result = run(command, f"{label}.attempt-{attempt}", log_dir, output_dir, expected_count)
        attempts.append(result)
        if result["returncode"] == 0:
            result["label"] = label
            result["attempts"] = attempt
            result["failed_attempt_seconds"] = sum(item["elapsed_seconds"] for item in attempts[:-1])
            return result
    result["label"] = label
    result["attempts"] = 3
    result["failed_attempt_seconds"] = sum(item["elapsed_seconds"] for item in attempts[:-1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--no-kernels", action="store_true")
    args = parser.parse_args()
    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    logs = args.benchmark_dir / "logs"
    logs.mkdir(exist_ok=True)
    inputs = sorted((args.benchmark_dir / "inputs").glob("*.yaml"))
    timings_path = args.benchmark_dir / "timings.json"
    prior = {}
    if timings_path.is_file():
        prior = {item["label"]: item for item in json.loads(timings_path.read_text())}
    common = ["boltz", "predict", "--cache", str(args.cache), "--use_msa_server", "--seed", "0"]
    if args.no_kernels:
        common.append("--no_kernels")
    measurements = []
    for input_path in inputs:
        out_dir = args.benchmark_dir / "single_runs" / input_path.stem
        if len(list(out_dir.rglob("affinity_*.json"))) == 1 and input_path.stem in prior:
            result = prior[input_path.stem]
            result["affinity_json_count"] = 1
        else:
            command = common[:2] + [str(input_path), "--out_dir", str(out_dir)] + common[2:]
            result = run_with_retries(command, input_path.stem, logs, out_dir, 1)
        measurements.append(result)
        timings_path.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    batch_out = args.benchmark_dir / "batch_run"
    if len(list(batch_out.rglob("affinity_*.json"))) == len(inputs) and "batch-09" in prior:
        batch_result = prior["batch-09"]
        batch_result["affinity_json_count"] = len(inputs)
    else:
        command = common[:2] + [str(args.benchmark_dir / "inputs"), "--out_dir", str(batch_out)] + common[2:]
        batch_result = run_with_retries(command, "batch-09", logs, batch_out, len(inputs))
    measurements.append(batch_result)
    timings_path.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    if any(item["returncode"] for item in measurements):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
