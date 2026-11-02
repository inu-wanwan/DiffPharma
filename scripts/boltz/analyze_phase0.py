"""Create the Phase-0 timing table, figure, and oracle budget memo."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--docking-timings", type=Path, required=True)
    parser.add_argument("--gpu-hours", type=float, default=6.0)
    parser.add_argument("--reserve-fraction", type=float, default=0.2)
    parser.add_argument(
        "--decode-seconds",
        type=float,
        default=21.0,
        help="Per-molecule DiffPharma decode cost included in the end-to-end budget",
    )
    args = parser.parse_args()
    manifest = json.loads((args.benchmark_dir / "manifest.json").read_text())
    timings = json.loads((args.benchmark_dir / "timings.json").read_text())
    singles = timings[:-1]
    if len(singles) != len(manifest) or any(item["returncode"] for item in timings):
        raise RuntimeError("benchmark is incomplete or contains failed predictions")
    rows = []
    docking_by_label: dict[str, float] = {}
    if args.docking_timings:
        docking_timings = json.loads(args.docking_timings.read_text(encoding="utf-8-sig"))
        if len(docking_timings) != len(manifest):
            raise RuntimeError("docking benchmark does not match the Boltz sample count")
        for item in docking_timings:
            if item["returncode"] or not item.get("output_exists", False):
                raise RuntimeError(f"docking failed for {item['label']}")
            docking_by_label[item["label"]] = item["elapsed_seconds"]
    for record, timing in zip(manifest, singles):
        rows.append({
            "sample": timing["label"],
            "pdb_chain": f'{record["pdb_id"].upper()}:{record["chain"]}',
            "protein_residues": record["protein_residues"],
            "pocket_residues": record["pocket_residues"],
            "ligand_heavy_atoms": record["ligand_heavy_atoms"],
            "elapsed_seconds": round(timing["elapsed_seconds"], 3),
            "docking_seconds": round(docking_by_label[timing["label"]], 3) if docking_by_label else "",
        })
    values = [row["elapsed_seconds"] for row in rows]
    batch_seconds = timings[-1]["elapsed_seconds"]
    per_item_batch = batch_seconds / len(rows)
    p95 = percentile(values, 0.95)
    usable_seconds = args.gpu_hours * 3600 * (1 - args.reserve_fraction)
    conservative_budget = math.floor(usable_seconds / p95)
    batch_budget = math.floor(usable_seconds / per_item_batch)
    combined_single_seconds = p95 + args.decode_seconds
    combined_batch_seconds = per_item_batch + args.decode_seconds
    combined_conservative_budget = math.floor(usable_seconds / combined_single_seconds)
    combined_batch_budget = math.floor(usable_seconds / combined_batch_seconds)
    recommended_budget = (combined_batch_budget // 50) * 50
    summary = {
        "n": len(values),
        "single_median_seconds": statistics.median(values),
        "single_p90_seconds": percentile(values, 0.90),
        "single_p95_seconds": p95,
        "single_max_seconds": max(values),
        "batch_total_seconds": batch_seconds,
        "batch_per_item_seconds": per_item_batch,
        "batch_speedup": statistics.median(values) / per_item_batch,
        "gpu_hours_per_experiment": args.gpu_hours,
        "reserve_fraction": args.reserve_fraction,
        "conservative_oracle_budget": conservative_budget,
        "batch_oracle_budget": batch_budget,
        "decode_seconds_per_item": args.decode_seconds,
        "combined_single_p95_seconds": combined_single_seconds,
        "combined_batch_per_item_seconds": combined_batch_seconds,
        "combined_conservative_budget": combined_conservative_budget,
        "combined_batch_budget": combined_batch_budget,
        "recommended_evaluation_cap": recommended_budget,
    }
    if docking_by_label:
        docking_values = [docking_by_label[row["sample"]] for row in rows]
        docking_median = statistics.median(docking_values)
        summary.update({
            "docking_median_seconds": docking_median,
            "docking_p95_seconds": percentile(docking_values, 0.95),
            "docking_max_seconds": max(docking_values),
            "boltz_batch_to_docking_median_ratio": per_item_batch / docking_median,
            "boltz_single_to_docking_median_ratio": statistics.median(values) / docking_median,
        })
    (args.benchmark_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.benchmark_dir / "timing_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, (ax, comparison_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [row["ligand_heavy_atoms"] for row in rows]
    scatter = ax.scatter([row["protein_residues"] for row in rows], values, c=colors, s=70, cmap="viridis")
    ax.axhline(statistics.median(values), color="tab:red", linestyle="--", label="single median")
    ax.axhline(per_item_batch, color="tab:blue", linestyle=":", label="batch per item")
    ax.set(xlabel="Protein residues", ylabel="End-to-end seconds", title="Boltz-2 Phase-0 throughput (RTX 4090)")
    fig.colorbar(scatter, ax=ax, label="Ligand heavy atoms")
    ax.legend()
    comparison_ax.bar(
        ["Vina-GPU+", "Boltz-2\nbatch/item", "Boltz-2\nsingle"],
        [summary["docking_median_seconds"], per_item_batch, statistics.median(values)]
        if docking_by_label else [0, per_item_batch, statistics.median(values)],
        color=["tab:gray", "tab:blue", "tab:red"],
    )
    comparison_ax.set_yscale("log")
    comparison_ax.set(ylabel="Median seconds (log scale)", title="Repeated oracle evaluation")
    comparison_ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.benchmark_dir / "throughput.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
