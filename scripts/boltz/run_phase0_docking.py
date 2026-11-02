"""Benchmark the docking path used by analysis/docking_GPU.py on Phase-0 inputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from rdkit import Chem


def checked(command: list[str], env: dict[str, str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    if completed.returncode:
        raise RuntimeError(f"command failed: {command}\n{completed.stderr[-2000:]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib:/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    measurements = []
    prepared = []
    for record in records:
        label = f"sample-{record['sample_index']:02d}"
        run_dir = args.workdir / label
        ligand_dir, output_dir = run_dir / "ligands", run_dir / "output"
        ligand_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        receptor = run_dir / "receptor.pdbqt"
        ligand = ligand_dir / "ligand.pdbqt"
        checked(["obabel", record["pocket_file"], "-O", str(receptor), "-xr"], env)
        checked(["obabel", record["ligand_file"], "-O", str(ligand)], env)
        molecule = next(m for m in Chem.SDMolSupplier(record["ligand_file"], removeHs=False) if m is not None)
        center = molecule.GetConformer().GetPositions().mean(axis=0)
        prepared.append({"label": label, "center": center.tolist(), "receptor": str(receptor), "ligand_dir": str(ligand_dir), "output_dir": str(output_dir)})
        (args.workdir / "prepared_inputs.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")
        if args.prepare_only:
            continue
        command = [
            str(args.binary), "--receptor", str(receptor), "--ligand_directory", str(ligand_dir),
            "--output_directory", str(output_dir), "--center_x", str(center[0]), "--center_y", str(center[1]),
            "--center_z", str(center[2]), "--size_x", "20", "--size_y", "20", "--size_z", "20",
            "--thread", "8000", "--search_depth", "16", "--seed", "0", "--opencl_binary_path", str(args.kernel_dir),
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, env=env)
        elapsed = time.perf_counter() - started
        (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        measurements.append({"label": label, "elapsed_seconds": elapsed, "returncode": completed.returncode})
        (args.workdir / "timings.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
