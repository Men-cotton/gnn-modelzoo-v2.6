#!/usr/bin/env python3
"""Recompute the numeric results reported in the IA3-2026 paper."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


LOG_TIMESTAMP = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})"
)
H100_ACCURACY = re.compile(
    r"\[Eval\] Step=(\d+), Wall=[\d.]+s, Val_Acc=([\d.]+)"
)
CS3_ACCURACY = re.compile(r"eval/masked_accuracy\s+=\s+([\d.]+)")
H100_THROUGHPUT = re.compile(r"\[Throughput\] Samples: ([\d.]+) samples/s")
CS3_TRAIN = re.compile(
    r"\| Train Device=CSX, Step=(\d+), Loss=[^,]+, "
    r"Rate=([\d.]+) samples/sec, GlobalRate=([\d.]+) samples/sec"
)
BATCH_SIZE = 4096
WIO_NORMALIZATION = 83


def final_accuracy(path: Path, system: str) -> float:
    pattern = H100_ACCURACY if system == "H100" else CS3_ACCURACY
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(match.lastindex or 1)))
    if not values:
        raise RuntimeError(f"No validation accuracy found in {path}")
    return values[-1]


def h100_throughput(path: Path) -> float:
    values = [
        float(match.group(1))
        for match in H100_THROUGHPUT.finditer(
            path.read_text(encoding="utf-8")
        )
    ]
    if not values:
        raise RuntimeError(f"No H100 throughput found in {path}")
    return values[-1]


def parse_timestamp(line: str) -> datetime | None:
    match = LOG_TIMESTAMP.match(line)
    if not match:
        return None
    return datetime.strptime(
        f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d %H:%M:%S.%f"
    )


def cs3_throughput(path: Path) -> float:
    points: list[tuple[datetime, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        timestamp = parse_timestamp(line)
        match = CS3_TRAIN.search(line)
        if timestamp is not None and match:
            points.append((timestamp, int(match.group(1))))
    if len(points) < 2:
        raise RuntimeError(f"Fewer than two CS-3 progress points in {path}")
    first, last = points[0], points[-1]
    step_delta = last[1] - first[1]
    seconds = (last[0] - first[0]).total_seconds()
    if step_delta <= 0 or seconds <= 0:
        raise RuntimeError(f"Invalid CS-3 progress window in {path}")
    return step_delta * BATCH_SIZE / seconds


def frame_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
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


def wio_fraction(path: Path) -> tuple[int, float]:
    manifest = json.loads(
        (path / "query_manifest.json").read_text(encoding="utf-8")
    )
    fractions: list[float] = []
    seen: set[tuple[Any, ...]] = set()
    for chunk in manifest["chunks"]:
        response = json.loads(
            (path / chunk["response"]).read_text(encoding="utf-8")
        )
        for row in frame_rows(response):
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            if row.get("replica_type") != "activation":
                continue
            if any(
                row.get(field) is None
                for field in ("timestamp", "replica_id", "it", "is")
            ):
                continue
            iteration_us = float(row["it"])
            input_empty_wio_us = float(row["is"])
            if iteration_us <= 0:
                continue
            iteration_s = iteration_us / 1_000_000
            input_empty_wio_s = input_empty_wio_us / 1_000_000
            fractions.append(
                input_empty_wio_s / iteration_s / WIO_NORMALIZATION
            )
    if not fractions:
        raise RuntimeError(f"No valid WIO records in {path}")
    return len(fractions), statistics.median(fractions)


def compute(root: Path) -> dict[str, Any]:
    logs = root / "logs"
    wio = root / "wio"

    accuracy = {
        "ogbn-arxiv": {
            "H100": final_accuracy(
                logs / "arxiv_graphsage_1gpu_accuracy_eval.log", "H100"
            ),
            "CS-3": final_accuracy(
                logs / "arxiv_graphsage_wse_accuracy_eval.log", "CS-3"
            ),
        },
        "ogbn-products": {
            "H100": final_accuracy(
                logs / "products_graphsage_1gpu_accuracy_eval.log", "H100"
            ),
            "CS-3": final_accuracy(
                logs / "products_graphsage_wse_accuracy_eval.log", "CS-3"
            ),
        },
    }

    throughput = {
        "ogbn-arxiv": {
            "cached": {
                "H100": h100_throughput(
                    logs / "arxiv_graphsage_1gpu_cache.log"
                ),
                "CS-3": cs3_throughput(
                    logs / "arxiv_graphsage_wse_cache.log"
                ),
            },
            "uncached": {
                "H100": h100_throughput(
                    logs / "arxiv_graphsage_1gpu_not.log"
                ),
                "CS-3": cs3_throughput(
                    logs / "arxiv_graphsage_wse_not.log"
                ),
            },
        },
        "ogbn-products": {
            "cached": {
                "H100": h100_throughput(
                    logs / "products_graphsage_1gpu_cache.log"
                ),
                "CS-3": cs3_throughput(
                    logs / "products_graphsage_wse_cache.log"
                ),
            },
            "uncached": {
                "H100": h100_throughput(
                    logs / "products_graphsage_1gpu_not.log"
                ),
                "CS-3": cs3_throughput(
                    logs / "products_graphsage_wse_not.log"
                ),
            },
        },
    }

    arxiv_rows, arxiv_fraction = wio_fraction(
        wio / "arxiv"
    )
    products_rows, products_fraction = wio_fraction(
        wio / "products"
    )
    return {
        "validation_accuracy": accuracy,
        "training_loop_throughput_samples_per_s": throughput,
        "normalized_empty_queue_fraction": {
            "ogbn-arxiv": {
                "records": arxiv_rows,
                "median": arxiv_fraction,
            },
            "ogbn-products": {
                "records": products_rows,
                "median": products_fraction,
            },
        },
    }


def at_reported_precision(actual: dict[str, Any]) -> dict[str, Any]:
    accuracy = {
        dataset: {
            system: round(value, 4)
            for system, value in systems.items()
        }
        for dataset, systems in actual["validation_accuracy"].items()
    }
    throughput = {
        dataset: {
            caching: {
                system: round(value)
                for system, value in systems.items()
            }
            for caching, systems in conditions.items()
        }
        for dataset, conditions in actual[
            "training_loop_throughput_samples_per_s"
        ].items()
    }
    normalized = {
        dataset: {
            "records": values["records"],
            "median_percent": round(values["median"] * 100, 1),
        }
        for dataset, values in actual[
            "normalized_empty_queue_fraction"
        ].items()
    }
    return {
        "validation_accuracy": accuracy,
        "training_loop_throughput_samples_per_s": throughput,
        "normalized_empty_queue_fraction": normalized,
    }


def compare_reported(actual: Any, reported: Any, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(reported, dict):
        if not isinstance(actual, dict):
            return [f"{path}: reported object, got {type(actual).__name__}"]
        for key, value in reported.items():
            child = f"{path}.{key}" if path else key
            if key not in actual:
                failures.append(f"{child}: missing")
            else:
                failures.extend(compare_reported(actual[key], value, child))
        return failures
    if isinstance(reported, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), reported, rel_tol=1e-12, abs_tol=5e-9
        ):
            failures.append(f"{path}: reported {reported!r}, got {actual!r}")
        return failures
    if actual != reported:
        failures.append(f"{path}: reported {reported!r}, got {actual!r}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare recomputed values with reported_metrics.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    actual = compute(artifact_root)
    print(json.dumps(actual, indent=2, sort_keys=True))
    if args.check:
        reported = json.loads(
            (artifact_root / "reported_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        failures = compare_reported(at_reported_precision(actual), reported)
        if failures:
            raise SystemExit("\n".join(failures))
        print("Reported metrics match.")


if __name__ == "__main__":
    main()
