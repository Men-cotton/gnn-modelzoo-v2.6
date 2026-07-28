import fnmatch
import glob
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class TrainingLogData:
    """Parsed training and evaluation metrics from one log file."""

    name: str

    # Validation / Evaluation Metrics
    eval_steps: List[int] = field(default_factory=list)
    eval_wall_times: List[float] = field(default_factory=list)
    eval_compute_times: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)

    # Training Throughput Metrics
    train_steps: List[int] = field(default_factory=list)
    train_wall_times: List[float] = field(default_factory=list)
    train_compute_times: List[float] = field(default_factory=list)
    local_throughputs: List[float] = field(default_factory=list)
    global_throughputs: List[float] = field(default_factory=list)
    summary_wall_time: Optional[float] = None

    # Per-step breakdown
    step_loads: List[float] = field(default_factory=list)
    step_preps: List[float] = field(default_factory=list)
    step_h2d_struc: List[float] = field(default_factory=list)
    step_h2d_fetch: List[float] = field(default_factory=list)
    step_h2ds: List[float] = field(default_factory=list)
    step_fwds: List[float] = field(default_factory=list)
    step_bwds: List[float] = field(default_factory=list)
    step_opts: List[float] = field(default_factory=list)

    def has_eval_data(self) -> bool:
        return len(self.eval_steps) > 0

    def has_train_data(self) -> bool:
        return len(self.train_steps) > 0


class LogPatterns:
    """Regex patterns for parsing GPU and CSX trainer logs."""

    LOG_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")

    PYG_MULTI_STEP = re.compile(r"\[Step=(\d+)\] Wall=([\d\.]+)s \| Loss=([\d\.]+)")
    PYG_MULTI_PROFILE = re.compile(
        r"\[Profile\] Avg ms/step \| Load: ([\d\.]+) "
        r"\(Prep: ([\d\.]+), Struc: ([\d\.]+), Fetch: ([\d\.]+)\) "
        r"\| Fwd: ([\d\.]+) \| Bwd: ([\d\.]+) \| Opt: ([\d\.]+) "
        r"\| GPU_Tot: ([\d\.]+)"
    )
    CSZOO_PROFILE = re.compile(
        r"\[Profile\] Avg ms/step \| Load: ([\d\.]+) "
        r"\| Host_Submit\(Fwd: ([\d\.]+), Bwd: ([\d\.]+), Opt: ([\d\.]+)\) "
        r"\| Residual\(Dev\): ([\d\.]+) \| Iter_Wall: ([\d\.]+)"
    )
    PYG_MULTI_THROUGHPUT = re.compile(
        r"\[Throughput\] Samples: ([\d\.]+) samples/s \(([\d\.]+)\)"
    )
    PYG_EVAL = re.compile(r"\[Eval\] Step=(\d+), Wall=([\d\.]+)s, Val_Acc=([\d\.]+)")
    WSE_EVAL_HEADER = re.compile(r"\| Eval Device=CSX, GlobalStep=(\d+),")
    WSE_EVAL_METRIC = re.compile(r"\s+-\s+eval/masked_accuracy\s+=\s+([\d\.]+)")
    WSE_TRAIN = re.compile(
        r"\| Train Device=CSX, Step=(\d+), Loss=[^,]+, "
        r"Rate=([\d\.]+) samples/sec, GlobalRate=([\d\.]+) samples/sec"
    )
    TRAIN_SUMMARY = re.compile(
        r"Processed \d+ training sample\(s\) in ([\d\.]+) seconds\."
    )


class LogParser:
    """Parses one log file into TrainingLogData."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.data = TrainingLogData(name=self.filename)

        self._current_step_data = {}
        self._current_compute_time = 0.0
        self._last_step = 0
        self._current_wse_eval_step = None
        self._wse_run_start: Optional[datetime] = None

    def parse(self) -> TrainingLogData:
        with open(self.filepath, "r") as f:
            for line in f:
                self._parse_line(line)
        return self.data

    @staticmethod
    def _parse_log_timestamp(line: str) -> Optional[datetime]:
        match = LogPatterns.LOG_TIMESTAMP.match(line)
        if not match:
            return None
        return datetime.strptime(f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d %H:%M:%S.%f")

    def _parse_line(self, line: str) -> bool:
        timestamp = self._parse_log_timestamp(line)
        if "Beginning appliance run" in line and timestamp is not None and self._wse_run_start is None:
            self._wse_run_start = timestamp
            return True

        match = LogPatterns.WSE_TRAIN.search(line)
        if match:
            step = int(match.group(1))
            wall_time = 0.0
            if timestamp is not None:
                if self._wse_run_start is None:
                    self._wse_run_start = timestamp
                wall_time = (timestamp - self._wse_run_start).total_seconds()

            self.data.train_steps.append(step)
            self.data.train_wall_times.append(wall_time)
            self.data.train_compute_times.append(0.0)
            self.data.local_throughputs.append(float(match.group(2)))
            self.data.global_throughputs.append(float(match.group(3)))
            return True

        match = LogPatterns.TRAIN_SUMMARY.search(line)
        if match:
            self.data.summary_wall_time = float(match.group(1))
            return True

        match = LogPatterns.PYG_EVAL.search(line)
        if match:
            self.data.eval_steps.append(int(match.group(1)))
            self.data.eval_wall_times.append(float(match.group(2)))
            self.data.accuracies.append(float(match.group(3)))
            self.data.eval_compute_times.append(0.0)
            return True

        match = LogPatterns.PYG_MULTI_STEP.search(line)
        if match:
            self._current_step_data = {
                "step": int(match.group(1)),
                "wall": float(match.group(2)),
            }
            return True

        match = LogPatterns.PYG_MULTI_PROFILE.search(line)
        if match and self._current_step_data:
            self._current_step_data["load"] = float(match.group(1)) / 1000.0
            self._current_step_data["prep"] = float(match.group(2)) / 1000.0
            self._current_step_data["struc"] = float(match.group(3)) / 1000.0
            self._current_step_data["fetch"] = float(match.group(4)) / 1000.0
            self._current_step_data["fwd"] = float(match.group(5)) / 1000.0
            self._current_step_data["bwd"] = float(match.group(6)) / 1000.0
            self._current_step_data["opt"] = float(match.group(7)) / 1000.0
            self._current_step_data["gpu_tot"] = float(match.group(8)) / 1000.0
            return True

        match = LogPatterns.CSZOO_PROFILE.search(line)
        if match and self._current_step_data:
            self._current_step_data["load"] = float(match.group(1)) / 1000.0
            self._current_step_data["prep"] = 0.0
            self._current_step_data["struc"] = 0.0
            self._current_step_data["fetch"] = 0.0
            self._current_step_data["fwd"] = float(match.group(2)) / 1000.0
            self._current_step_data["bwd"] = float(match.group(3)) / 1000.0
            self._current_step_data["opt"] = float(match.group(4)) / 1000.0
            self._current_step_data["gpu_tot"] = float(match.group(6)) / 1000.0
            return True

        match = LogPatterns.PYG_MULTI_THROUGHPUT.search(line)
        if match:
            self._flush_pyg_step(match)
            return True

        match = LogPatterns.WSE_EVAL_HEADER.search(line)
        if match:
            self._current_wse_eval_step = int(match.group(1))
            return True

        match = LogPatterns.WSE_EVAL_METRIC.search(line)
        if match and self._current_wse_eval_step is not None:
            self.data.eval_steps.append(self._current_wse_eval_step)
            self.data.eval_wall_times.append(0.0)
            self.data.eval_compute_times.append(0.0)
            self.data.accuracies.append(float(match.group(1)))
            self._current_wse_eval_step = None
            return True

        return False

    def _flush_pyg_step(self, throughput_match):
        if not self._current_step_data or "step" not in self._current_step_data:
            return

        d = self._current_step_data
        current_step = d["step"]

        delta_steps = current_step - self._last_step
        if delta_steps > 0:
            avg_gpu_time = d.get("gpu_tot", 0.0)
            self._current_compute_time += avg_gpu_time * delta_steps

        self._last_step = current_step

        self.data.train_steps.append(current_step)
        self.data.train_wall_times.append(d["wall"])
        self.data.step_loads.append(d.get("load", 0.0))
        self.data.step_preps.append(d.get("prep", 0.0))
        self.data.step_h2d_struc.append(d.get("struc", 0.0))
        self.data.step_h2d_fetch.append(d.get("fetch", 0.0))
        self.data.step_fwds.append(d.get("fwd", 0.0))
        self.data.step_bwds.append(d.get("bwd", 0.0))
        self.data.step_opts.append(d.get("opt", 0.0))
        self.data.train_compute_times.append(self._current_compute_time)
        self.data.global_throughputs.append(float(throughput_match.group(1)))
        self.data.local_throughputs.append(float(throughput_match.group(2)))
        self._current_step_data = {}


def load_training_logs(
    log_dir: str,
    include_log: Optional[List[str]] = None,
    exclude_log: Optional[List[str]] = None,
    sync_eval: bool = True,
) -> List[TrainingLogData]:
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"{log_dir} is not a directory.")

    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if include_log:
        log_files = [
            path for path in log_files
            if any(fnmatch.fnmatch(os.path.basename(path), pattern) for pattern in include_log)
        ]
    if exclude_log:
        log_files = [
            path for path in log_files
            if not any(fnmatch.fnmatch(os.path.basename(path), pattern) for pattern in exclude_log)
        ]
    if not log_files:
        raise FileNotFoundError(f"No .log files found in {log_dir}")

    print(f"Found {len(log_files)} log files.")

    all_data: List[TrainingLogData] = []
    for path in log_files:
        print(f"Parsing {os.path.basename(path)}...")
        data = LogParser(path).parse()
        if data.has_eval_data() or data.has_train_data():
            all_data.append(data)
        else:
            print(f"Warning: No valid metrics found in {data.name}")

    if not all_data:
        raise ValueError("No plotable data found.")

    if sync_eval:
        sync_eval_metrics(all_data)

    return all_data


def sync_eval_metrics(all_data: List[TrainingLogData]):
    """
    Replaces wall times and compute times in _eval logs with those from
    corresponding minimal logs. If no minimal log is found, clears wall/compute
    x-axis data for that eval log.
    """
    minimal_logs = {}
    for data in all_data:
        if not data.name.endswith("_eval.log") and "_eval" not in data.name:
            minimal_logs[data.name] = data

    for data in all_data:
        is_eval = data.name.endswith("_eval.log") or "_eval" in data.name
        if not is_eval:
            continue

        minimal_name = data.name.replace("_eval", "")
        target = minimal_logs.get(minimal_name)
        if target is None and data.has_train_data():
            target = data

        if target is None:
            print(f"Dropping Wall/Compute Time plot for {data.name} (no minimal log found).")
            data.eval_wall_times = []
            data.eval_compute_times = []
            continue

        if target is data:
            print(f"Aligning {data.name} with self (contains both train and eval)...")
        else:
            print(f"Aligning {data.name} with {target.name}...")

        step_to_wall = {s: w for s, w in zip(target.train_steps, target.train_wall_times)}
        step_to_compute = {s: c for s, c in zip(target.train_steps, target.train_compute_times)}

        data.eval_wall_times = [
            step_to_wall.get(step, data.eval_wall_times[i])
            for i, step in enumerate(data.eval_steps)
        ]
        data.eval_compute_times = [
            step_to_compute.get(step, data.eval_compute_times[i])
            for i, step in enumerate(data.eval_steps)
        ]
        print("  -> Replaced timestamps/compute times where possible.")
