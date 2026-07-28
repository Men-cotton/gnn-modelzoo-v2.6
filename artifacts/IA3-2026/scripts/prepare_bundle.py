#!/usr/bin/env python3
"""Build the IA3-2026 public artifact from the private evidence repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ARTIFACT_REVISION = "c834f2e511e869bfe32a1f3b602952da5674c3c4"
H100_RUN_REVISION = "8d1457914cfb8edfad40734500c663a483d13d90"

LOG_RECORDS = (
    {
        "result": "validation_accuracy",
        "dataset": "ogbn-arxiv",
        "system": "H100",
        "source": "arxiv_graphsage_1gpu_not_eval.log",
        "public": "arxiv_graphsage_1gpu_accuracy_eval.log",
        "run_id": None,
    },
    {
        "result": "validation_accuracy",
        "dataset": "ogbn-arxiv",
        "system": "CS-3",
        "source": "arxiv_graphsage_wse_accuracy_eval.log",
        "public": "arxiv_graphsage_wse_accuracy_eval.log",
        "run_id": "wsjob-mswfcbcppuzmiuapx9psr6",
    },
    {
        "result": "validation_accuracy",
        "dataset": "ogbn-products",
        "system": "H100",
        "source": "products_graphsage_1gpu_not_eval.log",
        "public": "products_graphsage_1gpu_accuracy_eval.log",
        "run_id": None,
    },
    {
        "result": "validation_accuracy",
        "dataset": "ogbn-products",
        "system": "CS-3",
        "source": "products_graphsage_wse_accuracy_eval.log",
        "public": "products_graphsage_wse_accuracy_eval.log",
        "run_id": "wsjob-ndve7gpbywsrd33jccuxnb",
    },
    {
        "result": "cached_throughput",
        "dataset": "ogbn-arxiv",
        "system": "H100",
        "source": "arxiv_graphsage_1gpu_cache.log",
        "public": "arxiv_graphsage_1gpu_cache.log",
        "run_id": None,
    },
    {
        "result": "cached_throughput",
        "dataset": "ogbn-arxiv",
        "system": "CS-3",
        "source": "arxiv_graphsage_wse_cache.log",
        "public": "arxiv_graphsage_wse_cache.log",
        "run_id": "wsjob-v7j6tkhq7nfcrdburs7sui",
    },
    {
        "result": "cached_throughput",
        "dataset": "ogbn-products",
        "system": "H100",
        "source": "products_graphsage_1gpu_cache.log",
        "public": "products_graphsage_1gpu_cache.log",
        "run_id": None,
    },
    {
        "result": "cached_throughput",
        "dataset": "ogbn-products",
        "system": "CS-3",
        "source": "products_graphsage_wse_cache.log",
        "public": "products_graphsage_wse_cache.log",
        "run_id": "wsjob-kqpue4uyt2s3wobgpkpw2c",
    },
    {
        "result": "uncached_throughput",
        "dataset": "ogbn-arxiv",
        "system": "H100",
        "source": "arxiv_graphsage_1gpu_not.log",
        "public": "arxiv_graphsage_1gpu_not.log",
        "run_id": None,
    },
    {
        "result": "uncached_throughput",
        "dataset": "ogbn-arxiv",
        "system": "CS-3",
        "source": "arxiv_graphsage_wse_not2.log",
        "public": "arxiv_graphsage_wse_not.log",
        "run_id": "wsjob-dw436fv2uoch2jqzzmfvco",
    },
    {
        "result": "uncached_throughput",
        "dataset": "ogbn-products",
        "system": "H100",
        "source": "products_graphsage_1gpu_not.log",
        "public": "products_graphsage_1gpu_not.log",
        "run_id": None,
    },
    {
        "result": "uncached_throughput",
        "dataset": "ogbn-products",
        "system": "CS-3",
        "source": "products_graphsage_wse_not.log",
        "public": "products_graphsage_wse_not.log",
        "run_id": "wsjob-s4ghbzereuamjqqdqtzg7w",
    },
)

CS3_CONFIGS = (
    (
        "arxiv_accuracy_trainer_params.yaml",
        "artifacts/model_dirs/ogbn_arxiv_graphsage_accuracy/"
        "20260624_190727/trainer_params.yaml",
    ),
    (
        "products_accuracy_trainer_params.yaml",
        "artifacts/model_dirs/ogbn_products_graphsage_accuracy/"
        "20260628_041034/trainer_params.yaml",
    ),
    (
        "arxiv_cached_throughput_trainer_params.yaml",
        "artifacts/model_dirs/ogbn_arxiv_graphsage_throughput_cache/"
        "20260624_141307/trainer_params.yaml",
    ),
    (
        "products_cached_throughput_trainer_params.yaml",
        "artifacts/model_dirs/ogbn_products_graphsage_throughput_cache/"
        "20260624_151908/trainer_params.yaml",
    ),
    (
        "arxiv_uncached_throughput_trainer_params.yaml",
        "artifacts/model_dirs/ogbn_arxiv_graphsage/cerebras_logs/"
        "20260616_142904/trainer_params.yaml",
    ),
    (
        "products_uncached_throughput_trainer_params.yaml",
        "artifacts/model_dirs/ogbn_products_graphsage/cerebras_logs/"
        "20260616_152437/trainer_params.yaml",
    ),
)

H100_CONFIGS = tuple(
    (
        f"{dataset}_{profile}.yaml",
        "src/cerebras/modelzoo/models/gnn/configs/"
        f"params_graphsage_ogbn_{dataset}_pyg_{source_profile}.yaml",
    )
    for dataset in ("arxiv", "products")
    for profile, source_profile in (
        ("accuracy", "accuracy_nocache"),
        ("cached_throughput", "throughput_cache"),
        ("uncached_throughput", "throughput_nocache"),
    )
)

WIO_RECORDS = (
    {
        "dataset": "ogbn-arxiv",
        "job_id": "wsjob-dw436fv2uoch2jqzzmfvco",
        "source_dir": "artifacts/grafana_dumps/20260617/"
        "wsjob-dw436fv2uoch2jqzzmfvco/raw_loki_chunked",
        "public_dir": "arxiv",
        "expected_raw_rows": 4824,
        "expected_activation_rows": 4221,
        "expected_complete_activation_rows": 4179,
    },
    {
        "dataset": "ogbn-products",
        "job_id": "wsjob-s4ghbzereuamjqqdqtzg7w",
        "source_dir": "artifacts/grafana_dumps/20260617/"
        "wsjob-s4ghbzereuamjqqdqtzg7w/raw_loki_chunked",
        "public_dir": "products",
        "expected_raw_rows": 9624,
        "expected_activation_rows": 8421,
        "expected_complete_activation_rows": 8400,
    },
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_text(text: str) -> str:
    text = text.replace(
        "/work/1/AC2/watanuki/gnn-modelzoo-v2.6", "$REPO_ROOT"
    )
    text = text.replace("/home/watanuki/gnn-modelzoo-v2.6", "$REPO_ROOT")
    text = re.sub(
        r"https://grafana\.anl0\.cerebras\.internal/\S+",
        "<internal-dashboard-url>",
        text,
    )
    text = re.sub(
        r"/n1/wsjob/workdir/[^\s,'\"]+",
        "<internal-workdir>",
        text,
    )
    text = re.sub(
        r"(?m)^(\[pbs\] PBS_JOBID=).*$",
        r"\1<redacted>",
        text,
    )
    text = re.sub(r"\bbnode\d+\b", "<compute-node>", text)
    text = text.replace("namespace: job-operator", "namespace: <internal-namespace>")
    text = text.replace("namespace=job-operator", "namespace=<internal-namespace>")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text


def write_sanitized(source: Path, target: Path) -> dict[str, str]:
    source_bytes = source.read_bytes()
    public_text = sanitize_text(source_bytes.decode("utf-8", errors="replace"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(public_text, encoding="utf-8")
    return {
        "source_sha256": sha256_bytes(source_bytes),
        "public_sha256": sha256_path(target),
    }


def git_file(repo: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def write_git_config(
    repo: Path, revision: str, source_path: str, target: Path
) -> dict[str, str]:
    source_bytes = git_file(repo, revision, source_path)
    public_text = sanitize_text(source_bytes.decode("utf-8", errors="replace"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(public_text, encoding="utf-8")
    return {
        "source_sha256": sha256_bytes(source_bytes),
        "public_sha256": sha256_path(target),
    }


def frame_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand every row in a Grafana DataFrame response without projection."""
    rows: list[dict[str, Any]] = []
    results = response.get("results")
    if not isinstance(results, dict):
        return rows

    for ref_id, result in results.items():
        frames = result.get("frames") if isinstance(result, dict) else None
        if not isinstance(frames, list):
            continue
        for frame_id, frame in enumerate(frames):
            fields = frame.get("schema", {}).get("fields", [])
            values = frame.get("data", {}).get("values", [])
            if not isinstance(values, list) or not values:
                continue
            names = [
                field.get("name")
                if isinstance(field, dict) and field.get("name")
                else f"field_{index}"
                for index, field in enumerate(fields)
            ]
            for raw_row in zip(*values):
                row = dict(zip(names, raw_row))
                labels = row.get("labels")
                if isinstance(labels, dict):
                    for key, value in labels.items():
                        row.setdefault(key, value)
                row.setdefault("refId", ref_id)
                row.setdefault("frame", frame_id)
                rows.append(row)
    return rows


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("id"),
        row.get("tsNs"),
        row.get("time"),
        row.get("Line"),
        row.get("pod"),
        row.get("message"),
    )


def canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def copy_wio_capture(
    source_dir: Path,
    target_dir: Path,
    expected_raw_rows: int,
    expected_activation_rows: int,
    expected_complete_activation_rows: int,
) -> dict[str, object]:
    """Copy the captured Grafana request/response JSON without modifying it."""
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    query = next(
        entry
        for entry in source_manifest["queries"]
        if entry["name"] == "02_rt_iter_perf_raw"
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    public_files: list[dict[str, object]] = []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    public_chunks: list[dict[str, object]] = []

    for chunk in query["chunks"]:
        public_chunk: dict[str, object] = {
            "from": chunk["from"],
            "to": chunk["to"],
        }
        for kind in ("request", "response"):
            source_path = source_dir / Path(chunk[kind]).name
            target_path = target_dir / source_path.name
            source_bytes = source_path.read_bytes()
            target_path.write_bytes(source_bytes)
            source_hash = sha256_bytes(source_bytes)
            public_hash = sha256_path(target_path)
            if source_hash != public_hash:
                raise RuntimeError(f"Byte-for-byte copy failed for {source_path}")
            public_chunk[kind] = target_path.name
            public_files.append(
                {
                    "path": target_path.name,
                    "sha256": public_hash,
                    "bytes": len(source_bytes),
                }
            )

        response = json.loads(
            (target_dir / str(public_chunk["response"])).read_text(
                encoding="utf-8"
            )
        )
        results = response.get("results")
        if not isinstance(results, dict) or not results or any(
            not isinstance(result, dict) or result.get("status") != 200
            for result in results.values()
        ):
            raise RuntimeError(f"{source_dir}: unsuccessful Grafana response")
        chunk_rows = frame_rows(response)
        added = 0
        for row in chunk_rows:
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            added += 1

        request = json.loads(
            (target_dir / str(public_chunk["request"])).read_text(
                encoding="utf-8"
            )
        )
        request_query = request["queries"][0]
        if request_query["expr"] != query["expr"]:
            raise RuntimeError(
                f"{source_dir}: request query differs from the capture manifest"
            )
        if request["from"] != chunk["from"] or request["to"] != chunk["to"]:
            raise RuntimeError(
                f"{source_dir}: request interval differs from the capture manifest"
            )
        max_lines = int(request_query["maxLines"])
        public_chunk.update(
            {
                "rows": len(chunk_rows),
                "deduplicated_rows_added": added,
                "max_lines": max_lines,
                "hit_max_lines": len(chunk_rows) >= max_lines,
            }
        )
        public_chunks.append(public_chunk)

    activation_rows = [
        row for row in rows if row.get("replica_type") == "activation"
    ]
    complete_activation_rows = [
        row
        for row in activation_rows
        if all(row.get(key) is not None for key in ("timestamp", "replica_id", "it", "is"))
    ]
    if len(rows) != expected_raw_rows:
        raise RuntimeError(
            f"{source_dir}: expected {expected_raw_rows} raw rows, found {len(rows)}"
        )
    if len(activation_rows) != expected_activation_rows:
        raise RuntimeError(
            f"{source_dir}: expected {expected_activation_rows} activation rows, "
            f"found {len(activation_rows)}"
        )
    if len(complete_activation_rows) != expected_complete_activation_rows:
        raise RuntimeError(
            f"{source_dir}: expected {expected_complete_activation_rows} complete "
            f"activation rows, found {len(complete_activation_rows)}"
        )
    if any(bool(chunk["hit_max_lines"]) for chunk in public_chunks):
        raise RuntimeError(f"{source_dir}: at least one chunk hit maxLines")

    reconstructed_jsonl = canonical_jsonl(rows)
    source_jsonl = source_dir / "02_rt_iter_perf_raw.jsonl"
    source_jsonl_hash = sha256_path(source_jsonl)
    reconstructed_hash = sha256_bytes(reconstructed_jsonl)
    if reconstructed_hash != source_jsonl_hash:
        raise RuntimeError(
            f"{source_dir}: response reconstruction does not match captured JSONL"
        )

    public_query_manifest = {
        "schema_version": 1,
        "capture_date": "2026-06-17",
        "wsjob": source_manifest["wsjob"],
        "from": source_manifest["from"],
        "to": source_manifest["to"],
        "loki_datasource_uid": source_manifest["loki_datasource_uid"],
        "query": query["expr"],
        "raw_rows": len(rows),
        "activation_rows": len(activation_rows),
        "complete_activation_rows": len(complete_activation_rows),
        "all_chunks_below_max_lines": True,
        "captured_jsonl_sha256": source_jsonl_hash,
        "reconstructed_jsonl_sha256": reconstructed_hash,
        "chunks": public_chunks,
        "files": sorted(public_files, key=lambda entry: str(entry["path"])),
    }
    query_manifest_path = target_dir / "query_manifest.json"
    query_manifest_path.write_text(
        json.dumps(public_query_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "source_dir": str(source_dir),
        "public_path": str(target_dir),
        "query_manifest": query_manifest_path.name,
        "query": query["expr"],
        "from": source_manifest["from"],
        "to": source_manifest["to"],
        "raw_rows": len(rows),
        "activation_rows": len(activation_rows),
        "complete_activation_rows": len(complete_activation_rows),
        "captured_jsonl_sha256": source_jsonl_hash,
        "reconstructed_jsonl_sha256": reconstructed_hash,
        "raw_files": len(public_files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wafer-root",
        type=Path,
        required=True,
        help="Private Wafer-GNN-manager checkout containing source evidence.",
    )
    parser.add_argument(
        "--gnn-repo",
        type=Path,
        required=True,
        help="gnn-modelzoo checkout containing the recorded H100 revision.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Artifact directory to populate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wafer_root = args.wafer_root.resolve()
    gnn_repo = args.gnn_repo.resolve()
    output_root = args.output_root.resolve()
    raw_log_root = wafer_root / "artifacts/raw_logs/2026-06-22"

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": "IA3-2026",
        "branch_base_revision": ARTIFACT_REVISION,
        "h100_run_revision": H100_RUN_REVISION,
        "sanitization": {
            "logs_and_configs": [
                "ALCF user and absolute repository paths",
                "compute-node hostnames and PBS job identifiers",
                "internal Cerebras dashboard URLs, workdirs, and namespace",
            ],
            "wio_request_response_chunks": (
                "none; copied byte for byte after credential and private-path scan"
            ),
        },
        "logs": [],
        "configs": [],
        "wio": {
            "allocated_activation_wios": 103,
            "saturatable_activation_wios": 83,
            "normalization": "input_empty_wio_us / 83 / iteration_us",
            "records": [],
        },
    }

    log_manifest = manifest["logs"]
    assert isinstance(log_manifest, list)
    for record in LOG_RECORDS:
        source = raw_log_root / str(record["source"])
        target = output_root / "logs" / str(record["public"])
        hashes = write_sanitized(source, target)
        log_manifest.append(
            {
                **record,
                "public_path": str(target.relative_to(output_root)),
                **hashes,
            }
        )

    config_manifest = manifest["configs"]
    assert isinstance(config_manifest, list)
    for public_name, source_rel in CS3_CONFIGS:
        source = wafer_root / source_rel
        target = output_root / "configs/cs3" / public_name
        hashes = write_sanitized(source, target)
        config_manifest.append(
            {
                "system": "CS-3",
                "source": source_rel,
                "public_path": str(target.relative_to(output_root)),
                **hashes,
            }
        )

    for public_name, source_path in H100_CONFIGS:
        target = output_root / "configs/h100" / public_name
        hashes = write_git_config(
            gnn_repo, H100_RUN_REVISION, source_path, target
        )
        config_manifest.append(
            {
                "system": "H100",
                "source_revision": H100_RUN_REVISION,
                "source": source_path,
                "public_path": str(target.relative_to(output_root)),
                **hashes,
            }
        )

    wio_manifest = manifest["wio"]
    assert isinstance(wio_manifest, dict)
    wio_records = wio_manifest["records"]
    assert isinstance(wio_records, list)
    for record in WIO_RECORDS:
        source = wafer_root / str(record["source_dir"])
        target = output_root / "wio" / str(record["public_dir"])
        copied = copy_wio_capture(
            source,
            target,
            int(record["expected_raw_rows"]),
            int(record["expected_activation_rows"]),
            int(record["expected_complete_activation_rows"]),
        )
        copied["source_dir"] = str(record["source_dir"])
        copied["public_path"] = str(target.relative_to(output_root))
        wio_records.append(
            {
                "dataset": record["dataset"],
                "job_id": record["job_id"],
                **copied,
            }
        )

    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
