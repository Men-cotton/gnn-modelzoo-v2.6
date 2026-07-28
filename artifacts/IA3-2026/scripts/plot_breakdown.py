import argparse
import os
from typing import List

import matplotlib.pyplot as plt

from log_parser import TrainingLogData, load_training_logs


def paper_breakdown_label(log_name: str) -> str:
    if log_name.startswith("arxiv_graphsage_"):
        dataset = "arxiv"
    elif log_name.startswith("products_graphsage_"):
        dataset = "products"
    else:
        dataset = "other"

    if "_cache" in log_name:
        cache = "cache"
    elif "_not" in log_name:
        cache = "no cache"
    else:
        return dataset

    return f"{dataset}, {cache}"


def add_log_filters(parser: argparse.ArgumentParser):
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


def plot_throughput_breakdown(all_data: List[TrainingLogData], output_file: str):
    names = []
    strucs = []
    fetchs = []
    fwds = []
    bwds = []
    opts = []
    cpu_overhead_idles = []

    for data in all_data:
        if "_eval" in data.name or not data.step_fwds:
            continue

        avg_fwd_time = sum(data.step_fwds) / len(data.step_fwds)
        if avg_fwd_time < 1e-6:
            print(f"Skipping breakdown for {data.name} (negligible forward time, likely WSE run).")
            continue

        fwd = avg_fwd_time
        bwd = sum(data.step_bwds) / len(data.step_bwds)
        opt = sum(data.step_opts) / len(data.step_opts)

        if data.step_h2d_struc and data.step_h2d_fetch:
            struc = sum(data.step_h2d_struc) / len(data.step_h2d_struc)
            fetch = sum(data.step_h2d_fetch) / len(data.step_h2d_fetch)
        elif data.step_h2ds:
            struc = sum(data.step_h2ds) / len(data.step_h2ds)
            fetch = 0.0
        elif data.step_loads:
            struc = sum(data.step_loads) / len(data.step_loads)
            fetch = 0.0
        else:
            struc = 0.0
            fetch = 0.0

        if len(data.train_wall_times) > 1:
            duration = data.train_wall_times[-1] - data.train_wall_times[0]
            steps = data.train_steps[-1] - data.train_steps[0]
            wall_per_step = duration / steps if steps > 0 else 0.0
        else:
            wall_per_step = 0.0

        gpu_total = struc + fetch + fwd + bwd + opt
        gap = max(0.0, wall_per_step - gpu_total) if wall_per_step > 0 else 0.0

        names.append(paper_breakdown_label(data.name))
        strucs.append(struc)
        fetchs.append(fetch)
        fwds.append(fwd)
        bwds.append(bwd)
        opts.append(opt)
        cpu_overhead_idles.append(gap)

    if not names:
        print("No valid breakdown data found for stacked bar chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    indices = range(len(names))
    width = 0.6

    ax.bar(indices, fwds, width, label="Forward")
    bottom_bwd = fwds
    ax.bar(indices, bwds, width, bottom=bottom_bwd, label="Backward")
    bottom_opt = [f + b for f, b in zip(fwds, bwds)]
    ax.bar(indices, opts, width, bottom=bottom_opt, label="Optimizer")
    bottom_struc = [f + b + o for f, b, o in zip(fwds, bwds, opts)]
    ax.bar(indices, strucs, width, bottom=bottom_struc, label="H2D (Struc/Load)")
    bottom_fetch = [f + b + o + s for f, b, o, s in zip(fwds, bwds, opts, strucs)]
    ax.bar(indices, fetchs, width, bottom=bottom_fetch, label="H2D (Fetch)")
    bottom_gap = [f + b + o + s + fe for f, b, o, s, fe in zip(fwds, bwds, opts, strucs, fetchs)]
    ax.bar(indices, cpu_overhead_idles, width, bottom=bottom_gap, label="CPU Overhead / Idle", hatch="//")

    ax.set_ylabel("Time per Step (s)")
    ax.set_title("Training Step Time Breakdown")
    ax.set_xticks(indices)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Breakdown plot saved to {output_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize training step breakdown from logs")
    parser.add_argument("log_dir", help="Path to the directory containing log files")
    parser.add_argument("--output", default="result.svg", help="Base output filename")
    add_log_filters(parser)
    args = parser.parse_args()

    all_data = load_training_logs(args.log_dir, args.include_log, args.exclude_log)
    base_name, ext = os.path.splitext(args.output)
    ext = ext if ext else ".svg"
    plot_throughput_breakdown(all_data, f"{base_name}_breakdown{ext}")


if __name__ == "__main__":
    main()
