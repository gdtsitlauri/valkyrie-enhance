from __future__ import annotations

from time import perf_counter

import numpy as np

from ..types import BenchmarkRecord, ResourceSnapshot


class BenchmarkSuiteModule:
    name = "benchmark_suite"

    def begin(self) -> float:
        return perf_counter()

    def end(self, started: float, frame: np.ndarray, resources: ResourceSnapshot) -> BenchmarkRecord:
        latency_ms = (perf_counter() - started) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        memory_gb = float(frame.nbytes) / (1024 ** 3)
        return BenchmarkRecord(
            fps=fps,
            latency_ms=latency_ms,
            gpu_load=resources.gpu_load,
            cpu_load=resources.cpu_load,
            memory_gb=memory_gb + resources.vram_used_gb,
        )
