#!/usr/bin/env python3
"""Emit PyG benchmark environment and effective config metadata."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pprint
import sys


def _print_python_environment(no_compile: bool) -> None:
    import torch
    import torch_geometric

    print("python_environment_begin")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
    print(f"torch_geometric={torch_geometric.__version__}")
    print(f"pyg_lib_available={importlib.util.find_spec('pyg_lib') is not None}")
    print(f"NO_COMPILE={os.environ.get('NO_COMPILE', '')}")
    print(f"requested_no_compile={no_compile}")
    print("python_environment_end")


def _print_effective_config(config_path: str) -> None:
    from cerebras.modelzoo.common.utils.run.config_loader import load_params_file

    cfg = load_params_file(config_path)
    summary = {
        "model": cfg["trainer"]["init"]["model"],
        "precision": cfg["trainer"]["init"].get("precision"),
        "loop": cfg["trainer"]["init"].get("loop"),
        "logging": cfg["trainer"]["init"].get("logging"),
        "train_dataloader": cfg["trainer"]["fit"]["train_dataloader"],
        "val_dataloader": cfg["trainer"]["validate"]["val_dataloader"],
    }
    print("effective_config_begin")
    pprint.pp(summary)
    print("effective_config_end")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    _print_python_environment(args.no_compile)
    _print_effective_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
