import argparse
import fnmatch
import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


LOG_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
PYG_THROUGHPUT = re.compile(r"\[Throughput\] Samples: ([\d.]+) samples/s")
WSE_TRAIN = re.compile(
    r"\| Train Device=CSX, Step=(\d+), Loss=[^,]+, "
    r"Rate=([\d.]+) samples/sec, GlobalRate=([\d.]+) samples/sec"
)


@dataclass
class WsePoint:
    timestamp: datetime
    step: int
    rate: float
    global_rate: float


@dataclass
class ThroughputRow:
    dataset: str
    accelerator: str
    caching: str
    throughput: float
    source: str
    window_steps: Optional[int] = None
    window_seconds: Optional[float] = None
    final_rate: Optional[float] = None
    final_global_rate: Optional[float] = None


def parse_timestamp(line: str) -> Optional[datetime]:
    match = LOG_TIMESTAMP.match(line)
    if not match:
        return None
    return datetime.strptime(
        f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d %H:%M:%S.%f"
    )


def parse_summary_key(log_name: str) -> Optional[Tuple[str, str, str]]:
    if "_eval" in log_name or "graphsage" not in log_name:
        return None

    if log_name.startswith("arxiv_graphsage_"):
        dataset = "arxiv"
    elif log_name.startswith("products_graphsage_"):
        dataset = "products"
    else:
        return None

    if "_1gpu_" in log_name:
        accelerator = "1gpu"
    elif "_wse_" in log_name:
        accelerator = "wse"
    else:
        return None

    if "_cache" in log_name:
        caching = "cache"
    elif "_not" in log_name:
        caching = "not"
    else:
        return None

    return dataset, accelerator, caching


def parse_wse_points(path: str) -> List[WsePoint]:
    points: List[WsePoint] = []
    with open(path, "r") as f:
        for line in f:
            timestamp = parse_timestamp(line)
            match = WSE_TRAIN.search(line)
            if match and timestamp is not None:
                points.append(
                    WsePoint(
                        timestamp=timestamp,
                        step=int(match.group(1)),
                        rate=float(match.group(2)),
                        global_rate=float(match.group(3)),
                    )
                )
    return points


def parse_pyg_final_throughput(path: str) -> Optional[float]:
    throughput = None
    with open(path, "r") as f:
        for line in f:
            match = PYG_THROUGHPUT.search(line)
            if match:
                throughput = float(match.group(1))
    return throughput


def wse_window_throughput(
    points: List[WsePoint],
    batch_size: int,
) -> Optional[Tuple[float, int, float]]:
    if len(points) < 2:
        return None

    first = points[0]
    last = points[-1]
    step_delta = last.step - first.step
    seconds = (last.timestamp - first.timestamp).total_seconds()
    if step_delta <= 0 or seconds <= 0:
        return None

    return step_delta * batch_size / seconds, step_delta, seconds


def parse_log(path: str, batch_size: int) -> Optional[ThroughputRow]:
    key = parse_summary_key(os.path.basename(path))
    if key is None:
        return None

    dataset, accelerator, caching = key
    if accelerator == "wse":
        points = parse_wse_points(path)
        result = wse_window_throughput(points, batch_size)
        if result is None:
            return None
        throughput, window_steps, window_seconds = result
        return ThroughputRow(
            dataset=dataset,
            accelerator=accelerator,
            caching=caching,
            throughput=throughput,
            source="wse_log_window",
            window_steps=window_steps,
            window_seconds=window_seconds,
            final_rate=points[-1].rate,
            final_global_rate=points[-1].global_rate,
        )

    throughput = parse_pyg_final_throughput(path)
    if throughput is None:
        return None
    return ThroughputRow(
        dataset=dataset,
        accelerator=accelerator,
        caching=caching,
        throughput=throughput,
        source="pyg_cumulative_throughput",
    )


def load_rows(
    log_dir: str,
    batch_size: int,
    include_log: List[str],
    exclude_log: List[str],
) -> Dict[Tuple[str, str, str], ThroughputRow]:
    paths = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if include_log:
        paths = [
            path
            for path in paths
            if any(fnmatch.fnmatch(os.path.basename(path), pattern) for pattern in include_log)
        ]
    if exclude_log:
        paths = [
            path
            for path in paths
            if not any(fnmatch.fnmatch(os.path.basename(path), pattern) for pattern in exclude_log)
        ]

    rows: Dict[Tuple[str, str, str], ThroughputRow] = {}
    for path in paths:
        row = parse_log(path, batch_size)
        if row is None:
            continue
        key = (row.dataset, row.accelerator, row.caching)
        old = rows.get(key)
        old_window = (old.window_steps or 0) if old is not None else 0
        new_window = row.window_steps or 0
        if old is None or new_window > old_window:
            rows[key] = row
    return rows


def format_int(value: float) -> str:
    return f"{round(value):,}"


def format_optional(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.2f}"


def format_optional_seconds(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.3f}"


def print_summary(rows: Dict[Tuple[str, str, str], ThroughputRow]) -> None:
    print("# dataset,cache,h100_throughput,cs3_throughput,ratio")
    for dataset in ("arxiv", "products"):
        for caching in ("cache", "not"):
            gpu = rows.get((dataset, "1gpu", caching))
            wse = rows.get((dataset, "wse", caching))
            if gpu is None or wse is None:
                continue
            cache_label = "Yes" if caching == "cache" else "No"
            ratio = wse.throughput / gpu.throughput if gpu.throughput else 0.0
            print(
                f"{dataset},{cache_label},{format_int(gpu.throughput)},"
                f"{format_int(wse.throughput)},{ratio:.4f}x"
            )


def print_detail(rows: Dict[Tuple[str, str, str], ThroughputRow]) -> None:
    print(
        "dataset\taccelerator\tcaching\tthroughput_samples_per_s\tsource\t"
        "window_steps\twindow_seconds\tfinal_rate\tfinal_global_rate"
    )
    for dataset in ("arxiv", "products"):
        for accelerator in ("1gpu", "wse"):
            for caching in ("cache", "not"):
                row = rows.get((dataset, accelerator, caching))
                if row is None:
                    continue
                print(
                    f"{dataset}\t{accelerator}\t{caching}\t"
                    f"{row.throughput:.2f}\t{row.source}\t"
                    f"{row.window_steps or ''}\t"
                    f"{format_optional_seconds(row.window_seconds)}\t"
                    f"{format_optional(row.final_rate)}\t"
                    f"{format_optional(row.final_global_rate)}"
                )


def add_log_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include-log",
        action="append",
        default=[],
        help="Include only log basenames matching this shell-style pattern. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-log",
        action="append",
        default=[],
        help="Exclude log basenames matching this shell-style pattern. Can be repeated.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute manuscript throughput values from GraphSAGE training logs."
    )
    parser.add_argument("log_dir", help="Path to a directory containing training logs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="CS-3 training batch size used to convert step deltas to sample counts.",
    )
    parser.add_argument(
        "--format",
        choices=("summary", "detail"),
        default="summary",
        help="Output format.",
    )
    add_log_filters(parser)
    args = parser.parse_args()

    rows = load_rows(args.log_dir, args.batch_size, args.include_log, args.exclude_log)
    if args.format == "summary":
        print_summary(rows)
    else:
        print_detail(rows)


if __name__ == "__main__":
    main()
