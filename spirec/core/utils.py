import csv
import logging
import os
import random
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import psutil
import torch


def setup_logger(name: str = "spirec") -> logging.Logger:
    """Configures structured console logging."""
    logger_inst = logging.getLogger(name)
    logger_inst.setLevel(logging.DEBUG)

    if not logger_inst.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger_inst.addHandler(handler)

    return logger_inst


logger = setup_logger()


def set_global_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """Enforces multi-library reproducibility across Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Reproducibility seed={seed} (Strict Determinism Enabled)")
    else:
        logger.info(f"Reproducibility seed={seed} (Benchmark Performance Enabled)")


class PipelineTimer:
    """Context manager for profiling wall-clock execution time."""

    def __init__(self, description: str, active: bool = True):
        self.description = description
        self.active = active
        self.start_time: float = 0.0

    def __enter__(self):
        if self.active:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active:
            elapsed = time.perf_counter() - self.start_time
            logger.info(f"TIMER: {self.description} execution: {elapsed:.3f} s")


class InferenceProfiler:
    """Monitors peak GPU VRAM allocation and latency per inference pass."""

    def __init__(self, model_identifier: str, destination_dir: str):
        self.model_name = model_identifier
        self.output_csv = os.path.join(destination_dir, "inference_performance.csv")
        self.clock_start: float = 0.0
        os.makedirs(destination_dir, exist_ok=True)

        if not os.path.exists(self.output_csv):
            with open(self.output_csv, "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Model", "Volume_ID", "Time_Seconds", "Peak_VRAM_GB"])

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.clock_start = time.perf_counter()

    def stop_and_record(self, volume_id: str) -> None:
        elapsed = time.perf_counter() - self.clock_start
        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        with open(self.output_csv, "a", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([self.model_name, volume_id, f"{elapsed:.3f}", f"{peak_vram:.3f}"])

        logger.info(
            f"[{self.model_name}] Vol: {volume_id} | Elapsed: {elapsed:.2f}s | Peak VRAM: {peak_vram:.2f} GB"
        )


class MemoryTelemetry:
    """Tracks physical host RAM and accelerator VRAM."""

    def __init__(self, log_path: str = "memory_telemetry.csv"):
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Event", "Index", "System_RAM_GB", "GPU_VRAM_GB"])

    def log(self, event_label: str, iteration_idx: int) -> None:
        ram_gb = psutil.virtual_memory().used / (1024**3)
        vram_gb = -1.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / (1024**3)

        with open(self.log_path, "a", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([event_label, iteration_idx, f"{ram_gb:.2f}", f"{vram_gb:.2f}"])

