"""
Render a combined low-level communication view from kernel_graph.json and wio_report.txt.

The figure overlays kernel graph PE endpoints on the wafer/fabric layout parsed from
the WIO report, and adds compact summaries for WIO edge usage and kernel routes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
    from matplotlib.gridspec import GridSpec  # noqa: E402  pylint: disable=wrong-import-position
    from matplotlib.patches import Patch, Rectangle  # noqa: E402  pylint: disable=wrong-import-position
except ImportError:  # pragma: no cover - handled at runtime
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    GridSpec = None  # type: ignore[assignment]
    Patch = None  # type: ignore[assignment]
    Rectangle = None  # type: ignore[assignment]

try:
    from analyze.WIO.wio_summary_visualizer import (
        FLOW_COLORS,
        WioReport,
        parse_wio_report,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from analyze.WIO.wio_summary_visualizer import (
        FLOW_COLORS,
        WioReport,
        parse_wio_report,
    )


KERNEL_FLOW_COLORS: Dict[str, str] = {
    **FLOW_COLORS,
    "IO": "#8c564b",
    "CTX": "#bcbd22",
    "OTHER": "#4a4a4a",
}


@dataclass(frozen=True)
class KernelPoint:
    x: int
    y: int
    flow: str
    endpoint: str
    route: Tuple[str, str]


@dataclass
class KernelSummary:
    edge_count: int
    points: List[KernelPoint]
    unique_pes: int
    flow_counts: Counter[str]
    route_counts: Counter[Tuple[str, str]]
    route_pe_counts: Counter[Tuple[str, str]]


def _classify_flow(*parts: str) -> str:
    text = " ".join(part.lower() for part in parts if part)
    checks = (
        ("ACT", ("act",)),
        ("EVT", ("evt", "event")),
        ("WGT", ("wgt", "weight", "wt")),
        ("GRD", ("grd", "grad")),
        ("DBG", ("dbg", "debug")),
        ("CTX", ("ctx", "context")),
        ("CMD", ("cmd", "command")),
        ("IO", ("io", "wio")),
    )
    for flow, tokens in checks:
        if any(token in text for token in tokens):
            return flow
    return "OTHER"


def _iter_port_points(edge: Dict[str, Any], endpoint: str) -> Iterable[KernelPoint]:
    pe_key = f"{endpoint}_port_pes"
    port_name = edge.get(f"{endpoint}_port_name", "")
    port_color = edge.get(f"{endpoint}_port_color_name", "")
    flow = _classify_flow(edge.get("key", ""), port_name, port_color)
    route = (edge.get("source_name", ""), edge.get("target_name", ""))

    for item in edge.get(pe_key, []):
        pe = item.get("pe") if isinstance(item, dict) else None
        if not pe:
            continue
        x = pe.get("x")
        y = pe.get("y")
        if isinstance(x, int) and isinstance(y, int):
            yield KernelPoint(x=x, y=y, flow=flow, endpoint=endpoint, route=route)


def load_kernel_summary(path: Path) -> KernelSummary:
    data = json.loads(path.read_text())
    edges = data.get("edges", [])
    points: List[KernelPoint] = []
    flow_counts: Counter[str] = Counter()
    route_counts: Counter[Tuple[str, str]] = Counter()
    route_pe_counts: Counter[Tuple[str, str]] = Counter()
    unique_pes: set[Tuple[int, int]] = set()

    for edge in edges:
        flow = _classify_flow(
            edge.get("key", ""),
            edge.get("source_port_name", ""),
            edge.get("target_port_name", ""),
            edge.get("source_port_color_name", ""),
            edge.get("target_port_color_name", ""),
        )
        route = (edge.get("source_name", ""), edge.get("target_name", ""))
        route_counts[route] += 1
        flow_counts[flow] += 1

        before = len(points)
        points.extend(_iter_port_points(edge, "source"))
        points.extend(_iter_port_points(edge, "target"))
        route_pe_counts[route] += len(points) - before

    for point in points:
        unique_pes.add((point.x, point.y))

    return KernelSummary(
        edge_count=len(edges),
        points=points,
        unique_pes=len(unique_pes),
        flow_counts=flow_counts,
        route_counts=route_counts,
        route_pe_counts=route_pe_counts,
    )


def _sample_points(points: List[KernelPoint], max_points: int) -> List[KernelPoint]:
    if max_points <= 0 or len(points) <= max_points:
        return points

    grouped: Dict[str, List[KernelPoint]] = defaultdict(list)
    for point in points:
        grouped[point.flow].append(point)

    sampled: List[KernelPoint] = []
    total = len(points)
    for flow_points in grouped.values():
        flow_limit = max(100, math.ceil(max_points * len(flow_points) / total))
        if len(flow_points) <= flow_limit:
            sampled.extend(flow_points)
            continue
        step = math.ceil(len(flow_points) / flow_limit)
        sampled.extend(flow_points[::step][:flow_limit])
    return sampled[:max_points]


def _short_name(name: str) -> str:
    if not name:
        return "<unknown>"
    parts = [part for part in name.split("/") if part]
    if len(parts) <= 3:
        return name
    return "/".join(parts[-3:])


def _draw_fabric(ax: Any, report: WioReport) -> None:
    fabric_w, fabric_h = report.fabric_columns, report.fabric_rows
    ax.add_patch(
        Rectangle(
            (0, 0),
            fabric_w,
            fabric_h,
            fill=False,
            edgecolor="#333333",
            linewidth=1.2,
        )
    )

    if report.buffer_columns:
        ax.add_patch(
            Rectangle(
                (0, 0),
                report.buffer_columns,
                fabric_h,
                facecolor="#efe5f8",
                edgecolor="none",
                alpha=0.45,
            )
        )
        ax.add_patch(
            Rectangle(
                (fabric_w - report.buffer_columns, 0),
                report.buffer_columns,
                fabric_h,
                facecolor="#efe5f8",
                edgecolor="none",
                alpha=0.45,
            )
        )

    if report.buffer_rows_below_core:
        start_y = report.compute_core_origin[1] + report.compute_core_size[1]
        ax.add_patch(
            Rectangle(
                (0, start_y),
                fabric_w,
                report.buffer_rows_below_core,
                facecolor="#fdebd0",
                edgecolor="none",
                alpha=0.55,
            )
        )

    ax.add_patch(
        Rectangle(
            report.compute_core_origin,
            report.compute_core_size[0],
            report.compute_core_size[1],
            facecolor="#d9ecf7",
            edgecolor="#1f77b4",
            linewidth=1.4,
            alpha=0.65,
        )
    )


def _draw_kernel_points(
    ax: Any, kernel: KernelSummary, max_kernel_points: int
) -> int:
    plotted = _sample_points(kernel.points, max_kernel_points)
    grouped: Dict[str, List[KernelPoint]] = defaultdict(list)
    for point in plotted:
        grouped[point.flow].append(point)

    for flow, points in sorted(grouped.items()):
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        ax.scatter(
            xs,
            ys,
            s=5,
            marker=".",
            color=KERNEL_FLOW_COLORS.get(flow, KERNEL_FLOW_COLORS["OTHER"]),
            alpha=0.28,
            linewidths=0,
            label=f"kernel {flow}",
        )
    return len(plotted)


def _draw_wio_points(ax: Any, report: WioReport) -> None:
    for placement in report.placements:
        y_pos = (placement.y_start + placement.y_end) / 2.0
        marker = "<" if placement.edge == "left" else ">"
        ax.scatter(
            placement.x,
            y_pos,
            s=42,
            marker=marker,
            color=FLOW_COLORS.get(placement.flow, "#1f77b4"),
            edgecolors="#111111",
            linewidths=0.35,
            alpha=0.95,
            zorder=4,
        )


def _draw_wio_bar(ax: Any, report: WioReport) -> None:
    flow_order = sorted(report.flows.keys())
    edges = ["left", "right"]
    y_positions = [0, 1]
    bottoms = [0, 0]
    counts: Dict[str, Counter[str]] = {"left": Counter(), "right": Counter()}
    for placement in report.placements:
        counts[placement.edge][placement.flow] += 1

    for flow in flow_order:
        values = [counts[edge].get(flow, 0) for edge in edges]
        ax.barh(
            y_positions,
            values,
            left=bottoms,
            color=FLOW_COLORS.get(flow, "#1f77b4"),
            edgecolor="#222222",
            linewidth=0.4,
            label=flow,
        )
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]

    ax.set_yticks(y_positions, labels=["Left edge", "Right edge"])
    ax.set_xlabel("WIO count")
    ax.set_title("WIO placement by edge")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=8, ncol=max(1, min(3, len(flow_order))))


def _summary_lines(
    report: WioReport,
    kernel: KernelSummary,
    plotted_kernel_points: int,
    kernel_graph_path: Path,
    wio_report_path: Path,
    top_routes: int,
) -> List[str]:
    lines = [
        "Inputs",
        f"  kernel_graph: {kernel_graph_path.name}",
        f"  wio_report:   {wio_report_path.name}",
        "",
        "WIO",
        f"  total: {report.total_wios}/{report.total_capacity}",
        f"  left/right: {report.left_wios}/{report.right_wios}",
        f"  core: origin={report.compute_core_origin}, size={report.compute_core_size}",
        "",
        "Kernel graph",
        f"  edges: {kernel.edge_count}",
        f"  PE endpoints: {len(kernel.points)}",
        f"  unique PEs: {kernel.unique_pes}",
        f"  plotted endpoints: {plotted_kernel_points}",
        f"  flow edges: {dict(kernel.flow_counts)}",
        "",
        "Top kernel routes",
    ]
    for (source, target), edge_count in kernel.route_counts.most_common(top_routes):
        pe_count = kernel.route_pe_counts[(source, target)]
        lines.append(
            f"  {_short_name(source)} -> {_short_name(target)}"
            f"  edges={edge_count}, endpoints={pe_count}"
        )
    return lines


def render_communication_graph(
    kernel: KernelSummary,
    report: WioReport,
    kernel_graph_path: Path,
    wio_report_path: Path,
    output_path: Path,
    fmt: str,
    title: str,
    max_kernel_points: int,
    top_routes: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for rendering.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[2.15, 1.0],
        height_ratios=[0.9, 1.1],
        wspace=0.24,
        hspace=0.25,
    )
    ax_map = fig.add_subplot(grid[:, 0])
    ax_bar = fig.add_subplot(grid[0, 1])
    ax_summary = fig.add_subplot(grid[1, 1])

    _draw_fabric(ax_map, report)
    plotted_kernel_points = _draw_kernel_points(ax_map, kernel, max_kernel_points)
    _draw_wio_points(ax_map, report)

    ax_map.set_xlim(0, report.fabric_columns)
    ax_map.set_ylim(0, report.fabric_rows)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel(f"Fabric column (0..{report.fabric_columns})")
    ax_map.set_ylabel(f"Fabric row (0..{report.fabric_rows})")
    ax_map.set_title(title)
    ax_map.grid(alpha=0.18, linestyle="--", linewidth=0.5)

    legend_items = [
        Patch(facecolor="#d9ecf7", edgecolor="#1f77b4", label="Compute core"),
        Patch(facecolor="#efe5f8", edgecolor="none", label="Buffer columns"),
    ]
    for flow in sorted({point.flow for point in kernel.points}):
        legend_items.append(
            Patch(
                facecolor=KERNEL_FLOW_COLORS.get(flow, KERNEL_FLOW_COLORS["OTHER"]),
                edgecolor="none",
                label=f"Kernel {flow}",
            )
        )
    for flow in sorted(report.flows):
        legend_items.append(
            Patch(
                facecolor=FLOW_COLORS.get(flow, "#1f77b4"),
                edgecolor="#111111",
                label=f"WIO {flow}",
            )
        )
    ax_map.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=4,
        fontsize=8,
    )

    _draw_wio_bar(ax_bar, report)

    ax_summary.axis("off")
    lines = _summary_lines(
        report=report,
        kernel=kernel,
        plotted_kernel_points=plotted_kernel_points,
        kernel_graph_path=kernel_graph_path,
        wio_report_path=wio_report_path,
        top_routes=top_routes,
    )
    ax_summary.text(
        0,
        1,
        "\n".join(lines),
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=8.5,
    )

    fig.savefig(output_path, format=fmt, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Path]:
    compile_dir = Path(args.compile_dir) if args.compile_dir else None
    kernel_graph = Path(args.kernel_graph) if args.kernel_graph else None
    wio_report = Path(args.wio_report) if args.wio_report else None

    if compile_dir:
        kernel_graph = kernel_graph or compile_dir / "kernel_graph.json"
        wio_report = wio_report or compile_dir / "wio_report.txt"

    if kernel_graph is None or wio_report is None:
        raise SystemExit(
            "Specify --compile-dir, or specify both --kernel-graph and --wio-report."
        )
    if not kernel_graph.exists():
        raise SystemExit(f"kernel_graph.json not found: {kernel_graph}")
    if not wio_report.exists():
        raise SystemExit(f"wio_report.txt not found: {wio_report}")
    return kernel_graph, wio_report


def _default_title(kernel_graph_path: Path) -> str:
    cs_dir = kernel_graph_path.parent.name
    return f"Communication graph overlay ({cs_dir})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize kernel_graph.json plus wio_report.txt communication layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--compile-dir",
        help="Compile artifact directory containing kernel_graph.json and wio_report.txt.",
    )
    parser.add_argument("--kernel-graph", help="Path to kernel_graph.json.")
    parser.add_argument("--wio-report", help="Path to wio_report.txt.")
    parser.add_argument(
        "--output",
        "-o",
        default="analyze/figures/communication_graph.svg",
        help="Output image path.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "svg"],
        default="svg",
        help="Output image format.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title. Defaults to the compile artifact directory name.",
    )
    parser.add_argument(
        "--max-kernel-points",
        type=int,
        default=25000,
        help="Maximum kernel PE endpoints to draw. Use 0 to draw all endpoints.",
    )
    parser.add_argument(
        "--top-routes",
        type=int,
        default=8,
        help="Number of kernel source-target routes shown in the text summary.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a compact parsed summary to stdout.",
    )

    args = parser.parse_args()
    if plt is None:
        raise SystemExit(
            "matplotlib is required. For example: uv run --with matplotlib python "
            "analyze/kernel/communication_graph_visualizer.py ..."
        )

    kernel_graph_path, wio_report_path = _resolve_inputs(args)
    kernel = load_kernel_summary(kernel_graph_path)
    report = parse_wio_report(wio_report_path)
    output_path = Path(args.output)
    title = args.title or _default_title(kernel_graph_path)

    render_communication_graph(
        kernel=kernel,
        report=report,
        kernel_graph_path=kernel_graph_path,
        wio_report_path=wio_report_path,
        output_path=output_path,
        fmt=args.format,
        title=title,
        max_kernel_points=args.max_kernel_points,
        top_routes=args.top_routes,
    )

    if args.print_summary:
        print(f"kernel edges: {kernel.edge_count}")
        print(f"kernel endpoints: {len(kernel.points)}")
        print(f"kernel unique PEs: {kernel.unique_pes}")
        print(f"WIO total: {report.total_wios}/{report.total_capacity}")
        print(f"output: {output_path}")


if __name__ == "__main__":
    main()
