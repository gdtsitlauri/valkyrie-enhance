from __future__ import annotations

import os
import time

import torch

from .config import ValkyrieConfig
from .types import ResourceSnapshot


class ResourceMonitor:
    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config
        self._pynvml_handle = None
        self._last_time = time.perf_counter()
        self._frame_times: list[float] = []
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore[import]
            pynvml.nvmlInit()
            self._pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml_handle = None

    def _gpu_load_nvml(self) -> tuple[float, float]:
        try:
            import pynvml  # type: ignore[import]
            util = pynvml.nvmlDeviceGetUtilizationRates(self._pynvml_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._pynvml_handle)
            return util.gpu / 100.0, mem.used / (1024 ** 3)
        except Exception:
            return self._gpu_load_torch()

    def _gpu_load_torch(self) -> tuple[float, float]:
        if torch.cuda.is_available() and self.config.runtime.prefer_cuda:
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return min(vram / max(total, 1.0), 1.0), vram
        return 0.0, 0.0

    def _cpu_load(self) -> float:
        if hasattr(os, "getloadavg"):
            return min(os.getloadavg()[0] / max(os.cpu_count() or 1, 1), 1.0)
        return 0.25

    def record_frame(self) -> None:
        now = time.perf_counter()
        self._frame_times.append(now - self._last_time)
        self._last_time = now
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)

    def fps_estimate(self) -> float:
        if len(self._frame_times) < 2:
            return self.config.runtime.target_fps
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / max(avg_dt, 1e-6)

    def snapshot(self) -> ResourceSnapshot:
        if self._pynvml_handle is not None:
            gpu_load, vram_used_gb = self._gpu_load_nvml()
        else:
            gpu_load, vram_used_gb = self._gpu_load_torch()
        cpu_load = self._cpu_load()
        fps = self.fps_estimate()
        emergency = gpu_load > 0.95 or vram_used_gb > self.config.runtime.vram_budget_gb
        return ResourceSnapshot(
            gpu_load=gpu_load,
            cpu_load=cpu_load,
            vram_used_gb=vram_used_gb,
            fps_estimate=fps,
            emergency=emergency,
        )
