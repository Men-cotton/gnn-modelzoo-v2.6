import argparse
import os
from typing import List

import matplotlib.pyplot as plt

from log_parser import TrainingLogData, load_training_logs


def paper_platform_label(log_name: str) -> str:
    if "_1gpu_" in log_name:
        return "H100"
    if "_wse_" in log_name:
        return "CS-3"
    return "Other"


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


def plot_accuracy_set(all_data: List[TrainingLogData], output_file: str):
    valid_data = [data for data in all_data if data.has_eval_data() and len(data.eval_steps) > 1]
    if not valid_data:
        print("No valid accuracy data found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Accuracy vs Wall Time", "Accuracy vs Compute Time", "Accuracy vs Steps"]
    x_labels = ["Wall Time (s)", "Compute Time (s)", "Steps"]
    x_attrs = ("eval_wall_times", "eval_compute_times", "eval_steps")

    for i, ax in enumerate(axes):
        for data in valid_data:
            x_vals = getattr(data, x_attrs[i])
            y_vals = data.accuracies

            if i == 1:
                filtered = [(x, y) for x, y in zip(x_vals, y_vals) if x > 0]
                if filtered:
                    x_vals, y_vals = zip(*filtered)
                else:
                    x_vals, y_vals = [], []

            if x_vals and y_vals:
                ax.plot(x_vals, y_vals, marker=".", label=paper_platform_label(data.name))

        ax.set_xlim(left=0)
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel("Validation Accuracy")
        ax.set_title(titles[i])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close(fig)


def plot_accuracy_steps(all_data: List[TrainingLogData], output_file: str):
    valid_data = [data for data in all_data if data.has_eval_data() and len(data.eval_steps) > 1]
    if not valid_data:
        print("No valid accuracy data found for accuracy-vs-steps plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for data in valid_data:
        ax.plot(data.eval_steps, data.accuracies, marker=".", label=paper_platform_label(data.name))

    ax.set_xlim(left=0)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Accuracy vs Steps")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Accuracy steps plot saved to {output_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize validation accuracy from training logs")
    parser.add_argument("log_dir", help="Path to the directory containing log files")
    parser.add_argument("--output", default="result.svg", help="Base output filename")
    parser.add_argument(
        "--steps-only",
        action="store_true",
        help="Generate only the compact accuracy-vs-steps figure at --output.",
    )
    add_log_filters(parser)
    args = parser.parse_args()

    all_data = load_training_logs(args.log_dir, args.include_log, args.exclude_log)
    if args.steps_only:
        plot_accuracy_steps(all_data, args.output)
        return

    base_name, ext = os.path.splitext(args.output)
    ext = ext if ext else ".svg"
    plot_accuracy_set(all_data, f"{base_name}_accuracy{ext}")


if __name__ == "__main__":
    main()
