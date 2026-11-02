"""Freeze a stratified CrossDocked Phase-0 benchmark and write Boltz YAMLs."""
from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

import yaml
from rdkit import Chem


def _pocket_size(path: Path) -> int:
    residues = set()
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            residues.add((line[21:22], line[22:26].strip(), line[26:27]))
    return len(residues)


def _ligand(path: Path) -> tuple[str, int]:
    molecule = next((m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None), None)
    if molecule is None:
        raise ValueError(f"cannot read ligand: {path}")
    return Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True), molecule.GetNumHeavyAtoms()


def _sequence(pdb_id: str, chain: str) -> str:
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/display"
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode()
    entries = []
    header = None
    sequence: list[str] = []
    for line in text.splitlines() + [">"]:
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(sequence)))
            header, sequence = line[1:], []
        else:
            sequence.append(line.strip())
    # CrossDocked filenames use author chain IDs. RCSB FASTA headers may expose
    # those as e.g. ``Chains E[auth A], F[auth B]`` after mmCIF normalization.
    chain_tokens = (f"Chain {chain}", f"Chains {chain},", f", {chain}", f"[auth {chain}]")
    matches = [seq for hdr, seq in entries if any(token in hdr for token in chain_tokens)]
    if not matches:
        raise ValueError(f"RCSB FASTA has no chain {chain} for {pdb_id}")
    return max(matches, key=len)


def _select_grid(records: list[dict], count: int) -> list[dict]:
    """Greedily choose points nearest a 3x3 quantile grid without duplicates."""
    if count != 9:
        raise ValueError("the preregistered Phase-0 design uses exactly 9 complexes")
    pockets = sorted(r["pocket_residues"] for r in records)
    ligands = sorted(r["ligand_heavy_atoms"] for r in records)
    quantiles = (0.15, 0.5, 0.85)
    targets = [(pockets[round(qp * (len(pockets) - 1))], ligands[round(ql * (len(ligands) - 1))]) for qp in quantiles for ql in quantiles]
    selected: list[dict] = []
    used: set[str] = set()
    pspan = max(pockets) - min(pockets) or 1
    lspan = max(ligands) - min(ligands) or 1
    for target_p, target_l in targets:
        candidate = min(
            (r for r in records if r["target_id"] not in used),
            key=lambda r: ((r["pocket_residues"] - target_p) / pspan) ** 2 + ((r["ligand_heavy_atoms"] - target_l) / lspan) ** 2,
        )
        selected.append(candidate)
        used.add(candidate["target_id"])
    return selected


def prepare(dataset_dir: Path, output_dir: Path) -> None:
    records = []
    for pocket in sorted(dataset_dir.glob("*-pocket10.pdb")):
        prefix = pocket.name.removesuffix("-pocket10.pdb")
        parts = prefix.split("-")
        pdb_id, chain = parts[0], parts[1]
        sdf = next(iter(sorted(dataset_dir.glob(f"{pocket.stem}_*.sdf"))), None)
        if sdf is None:
            raise FileNotFoundError(f"no ligand SDF for {pocket.name}")
        smiles, heavy_atoms = _ligand(sdf)
        records.append({
            "target_id": prefix,
            "pdb_id": pdb_id,
            "chain": chain,
            "pocket_file": str(pocket),
            "ligand_file": str(sdf),
            "pocket_residues": _pocket_size(pocket),
            "ligand_heavy_atoms": heavy_atoms,
            "ligand_smiles": smiles,
        })
    selected = _select_grid(records, 9)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = output_dir / "inputs"
    inputs.mkdir(exist_ok=True)
    for index, record in enumerate(selected):
        record["sample_index"] = index
        record["protein_sequence"] = _sequence(record["pdb_id"], record["chain"])
        record["protein_residues"] = len(record["protein_sequence"])
        document = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": record["protein_sequence"]}},
                {"ligand": {"id": "B", "smiles": record["ligand_smiles"]}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        yaml_path = inputs / f"sample-{index:02d}.yaml"
        yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        record["boltz_input"] = str(yaml_path)
    (output_dir / "manifest.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k in selected[0] if k not in {"protein_sequence", "ligand_smiles"}])
        writer.writeheader()
        writer.writerows({k: v for k, v in record.items() if k in writer.fieldnames} for record in selected)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset_dir, args.output_dir)
