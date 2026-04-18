from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class ModuleStats:
    name: str
    elapsed_ms: float
    metadata: dict[str, Any]


class BaseModule:
    name = "base"

    def timed(self, fn, *args, **kwargs):
        started = perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (perf_counter() - started) * 1000.0
        return result, ModuleStats(self.name, elapsed_ms, {})
