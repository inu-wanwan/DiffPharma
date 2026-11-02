# Boltz-2 Phase 0 benchmark

Measured on 2026-11-02 on an NVIDIA GeForce RTX 4090 (24 GB, driver 610.88).
This is the frozen throughput baseline used to choose the Phase 5 oracle budget.

## Frozen conditions

- CrossDocked test split: 9 complexes selected before timing by the Cartesian
  product of the 15th, 50th, and 85th percentiles of pocket-residue count and
  ligand-heavy-atom count.
- Boltz: v2.2.1, commit `cb04aeccdd480fd4db707f0bbafde538397fa2ac`.
- Full protein chains were fetched from RCSB; ligands were submitted as
  canonical SMILES. The timed run did not add pocket-contact constraints.
- Default Boltz prediction settings, seed 0, `--no_kernels`, and ColabFold MSA
  server enabled. Timings include MSA requests but exclude the one-time model
  weight and CCD download.
- Single mode runs one YAML per process. Batch mode submits all 9 YAMLs to one
  Boltz process.
- Vina comparison follows the repeated-evaluation path in
  `analysis/docking_GPU.py`: Vina-GPU+ 2.0, cached OpenCL kernels, seed 0,
  20-Angstrom box, 8,000 threads, and search depth 16.

## Results

| Metric | Result |
|---|---:|
| Boltz single median | 62.555 s |
| Boltz single p90 / p95 / max | 74.015 / 75.514 / 77.014 s |
| Boltz batch total (9) | 126.821 s |
| Boltz batch per ligand | 14.091 s |
| Batch speedup vs single median | 4.439x |
| Vina-GPU+ median / p95 / max | 2.733 / 5.887 / 6.029 s |
| Boltz batch / Vina median | 5.156x |
| Boltz single / Vina median | 22.891x |

The per-complex values and the plotted size dependence are in
`timing_table.csv` and `throughput.png` in this directory.

## End-to-end oracle budget decision

The working allocation is **6 GPU-hours per optimization experiment**, with a
20% operational reserve. This allocation is a Phase 0 planning assumption, not
an externally imposed quota. Per the Phase 0 task note, the budget includes the
DiffPharma decode that precedes each Boltz call. Phase 1 measured 4.7 s/molecule
at NFE=100 and 21 s/molecule at NFE=500; the adopted cap uses the slower NFE=500
profile.

- Boltz-only batch capacity: floor(17,280 / 14.091) = 1,226 misses.
- NFE=100 end-to-end batch capacity: floor(17,280 / (14.091 + 4.7)) = 919.
- NFE=500 end-to-end batch capacity: floor(17,280 / (14.091 + 21)) = 492.
- Conservative NFE=500 single p95 capacity: floor(17,280 / (75.514 + 21)) = 179.
- Adopted cap: **450 decode + Boltz cache misses per experiment**.

The cap is enforced in `config/boltz.yml`; the client counts Boltz cache misses,
and the optimization driver must apply the same 450 cap to decode attempts.
Batch capacity is comfortably in the hundreds, so Phase 5 keeps
Boltz-2 as the optimization oracle rather than switching to a
docking/surrogate-only plan.

## Interpretation limits

This is a throughput and feasibility benchmark, not an affinity-accuracy
benchmark. Boltz rebuilt each complex from sequence and SMILES, while Vina used
the supplied pocket geometry, so their elapsed times describe the two actual
oracle paths but their structures and scores are not interchangeable. Before a
scientific Phase 5 comparison, keep this frozen set and settings, precompute MSA
for HPC use, and report the raw Boltz affinity heads alongside independent
docking scores.
