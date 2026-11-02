"""Process-boundary client for using Boltz-2 as an affinity oracle.

This module deliberately does not import :mod:`boltz`: DiffPharma and Boltz
require incompatible Python environments.  It writes Boltz YAML inputs, starts
the configured command, and reads its JSON outputs instead.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


@dataclasses.dataclass(frozen=True)
class BoltzRecord:
    target_id: str
    protein_sequence: str
    ligand_smiles: str
    pocket_contacts: tuple[tuple[str, int], ...] = ()
    record_id: str | None = None


@dataclasses.dataclass(frozen=True)
class BoltzResult:
    record: BoltzRecord
    cache_key: str
    affinity_pred_value: float | None = None
    affinity_probability_binary: float | None = None
    affinity_pred_value1: float | None = None
    affinity_pred_value2: float | None = None
    affinity_probability_binary1: float | None = None
    affinity_probability_binary2: float | None = None
    cached: bool = False
    error: str | None = None

    @property
    def optimization_value(self) -> float | None:
        """Return a maximise-able value; Boltz log10(uM) is lower-is-better."""
        return None if self.affinity_pred_value is None else -self.affinity_pred_value


class BudgetExhausted(RuntimeError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Boltz configuration must be a YAML mapping")
    return config


def _canonical_smiles(smiles: str) -> str:
    """Use RDKit when it is available, without making it a client dependency."""
    try:
        from rdkit import Chem  # type: ignore
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        return Chem.MolToSmiles(molecule, canonical=True)
    except ImportError:
        return smiles.strip()


def _inference_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "boltz_commit": config["boltz_commit"],
        "prediction": config.get("prediction", {}),
        "use_msa_server": config.get("use_msa_server", False),
        "msa_server_url": config.get("msa_server_url"),
    }


def cache_key(record: BoltzRecord, config: Mapping[str, Any]) -> str:
    payload = {
        "target_id": record.target_id,
        "canonical_smiles": _canonical_smiles(record.ligand_smiles),
        "parameters": _inference_parameters(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def affinity_reward(result: BoltzResult) -> float | None:
    """Boltz paper's optional combined reward, retaining raw heads in storage."""
    if result.affinity_pred_value1 is None or result.affinity_probability_binary1 is None:
        return None
    return max((-result.affinity_pred_value1 + 2.0) / 4.0, 0.0) * result.affinity_probability_binary1


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "scores" / key[:2] / f"{key}.json"


def _result_from_data(record: BoltzRecord, key: str, data: Mapping[str, Any], *, cached: bool) -> BoltzResult:
    fields = {field.name for field in dataclasses.fields(BoltzResult)}
    values = {name: data.get(name) for name in fields if name in data}
    return BoltzResult(record=record, cache_key=key, cached=cached, **values)


def _read_cache(record: BoltzRecord, key: str, cache_dir: Path) -> BoltzResult | None:
    path = _cache_path(cache_dir, key)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return _result_from_data(record, key, json.load(handle), cached=True)


def _write_cache(result: BoltzResult, cache_dir: Path) -> None:
    path = _cache_path(cache_dir, result.cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(result)
    payload.pop("record")
    payload.pop("cache_key")
    payload.pop("cached")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _reserve_budget(cache_dir: Path, count: int, maximum: int | None) -> None:
    if not count or maximum is None:
        return
    database = cache_dir / "budget.sqlite3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS counter (id INTEGER PRIMARY KEY CHECK (id = 1), used INTEGER NOT NULL)")
        connection.execute("INSERT OR IGNORE INTO counter (id, used) VALUES (1, 0)")
        used = connection.execute("SELECT used FROM counter WHERE id = 1").fetchone()[0]
        if used + count > maximum:
            raise BudgetExhausted(f"Boltz evaluation budget exhausted: {used}/{maximum}, requested {count}")
        connection.execute("UPDATE counter SET used = used + ? WHERE id = 1", (count,))


def _input_document(record: BoltzRecord) -> dict[str, Any]:
    document: dict[str, Any] = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": record.protein_sequence}},
            {"ligand": {"id": "B", "smiles": record.ligand_smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    if record.pocket_contacts:
        document["constraints"] = [{"pocket": {"binder": "B", "contacts": [[chain, residue] for chain, residue in record.pocket_contacts]}}]
    return document


def _command(config: Mapping[str, Any], inputs: Path, output: Path, cache: Path) -> list[str]:
    launcher = config.get("launcher", [])
    if isinstance(launcher, str):
        launcher = launcher.split()
    command = list(launcher) + [str(config.get("executable", "boltz")), "predict", str(inputs), "--out_dir", str(output), "--cache", str(cache)]
    prediction = config.get("prediction", {})
    for name, value in prediction.items():
        option = f"--{name}"
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None:
            command.extend((option, str(value)))
    if config.get("use_msa_server"):
        command.append("--use_msa_server")
        if config.get("msa_server_url"):
            command.extend(("--msa_server_url", str(config["msa_server_url"])))
    return command


def score_batch(records: Sequence[BoltzRecord], workdir: str | Path, config: Mapping[str, Any]) -> list[BoltzResult]:
    """Score records in one Boltz batch, returning one result per input record.

    Only cache misses consume the persistent oracle budget. Failed predictions
    are represented explicitly and are intentionally never cached as scores.
    """
    cache_dir = Path(str(config["cache_dir"])).expanduser()
    workdir = Path(workdir)
    keys = [cache_key(record, config) for record in records]
    results: list[BoltzResult | None] = [_read_cache(record, key, cache_dir) for record, key in zip(records, keys)]
    pending = [(index, record, key) for index, (record, key, result) in enumerate(zip(records, keys, results)) if result is None]
    if not pending:
        return [result for result in results if result is not None]

    _reserve_budget(cache_dir, len(pending), config.get("max_evaluations"))
    run_dir = workdir / f"boltz-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    inputs = run_dir / "inputs"
    output = run_dir / "output"
    inputs.mkdir(parents=True)
    identifiers: dict[int, str] = {}
    for index, record, _ in pending:
        identifier = record.record_id or f"record-{index}"
        if not identifier.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"record_id must be filename-safe: {identifier!r}")
        identifiers[index] = identifier
        with (inputs / f"{identifier}.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_input_document(record), handle, sort_keys=False)

    completed = subprocess.run(_command(config, inputs, output, cache_dir), text=True, capture_output=True, check=False)
    for index, record, key in pending:
        identifier = identifiers[index]
        affinity_file = output / "boltz_results_inputs" / "predictions" / identifier / f"affinity_{identifier}.json"
        if affinity_file.is_file():
            with affinity_file.open(encoding="utf-8") as handle:
                result = _result_from_data(record, key, json.load(handle), cached=False)
            _write_cache(result, cache_dir)
        else:
            error = f"Boltz produced no affinity JSON (exit {completed.returncode})"
            if completed.stderr:
                error += f": {completed.stderr.strip()[-1000:]}"
            result = BoltzResult(record=record, cache_key=key, error=error)
        results[index] = result
    return [result for result in results if result is not None]


def score(record: BoltzRecord, workdir: str | Path, config: Mapping[str, Any]) -> BoltzResult:
    return score_batch([record], workdir, config)[0]


def _record_from_json(data: Mapping[str, Any]) -> BoltzRecord:
    contacts = tuple((str(chain), int(residue)) for chain, residue in data.get("pocket_contacts", []))
    return BoltzRecord(target_id=data["target_id"], protein_sequence=data["protein_sequence"], ligand_smiles=data["ligand_smiles"], pocket_contacts=contacts, record_id=data.get("record_id"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Boltz affinity scoring over the process boundary")
    parser.add_argument("--config", type=Path, default=Path("config/boltz.yml"))
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    arguments = parser.parse_args()
    records = [_record_from_json(json.loads(line)) for line in arguments.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = score_batch(records, arguments.workdir, load_config(arguments.config))
    arguments.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output_jsonl.open("w", encoding="utf-8") as handle:
        for result in results:
            item = dataclasses.asdict(result)
            item["optimization_value"] = result.optimization_value
            item["affinity_reward"] = affinity_reward(result)
            handle.write(json.dumps(item, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
