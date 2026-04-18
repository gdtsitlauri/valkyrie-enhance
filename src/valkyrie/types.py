from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Frame = np.ndarray
AudioChunk = np.ndarray


@dataclass(slots=True)
class SceneAttributes:
    content_type: str = "unknown"
    motion_level: str = "medium"
    lighting: str = "balanced"
    has_faces: bool = False
    artifact_level: float = 0.0
    style: str = "generic"


@dataclass(slots=True)
class ResourceSnapshot:
    gpu_load: float = 0.0
    cpu_load: float = 0.0
    vram_used_gb: float = 0.0
    fps_estimate: float = 60.0
    emergency: bool = False


@dataclass(slots=True)
class ModuleDecision:
    enabled: dict[str, bool] = field(default_factory=dict)
    quality_multipliers: dict[str, float] = field(default_factory=dict)
    rationale: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkRecord:
    fps: float
    latency_ms: float
    gpu_load: float
    cpu_load: float
    memory_gb: float
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None


@dataclass(slots=True)
class PipelineResult:
    frame: Frame
    audio: AudioChunk | None
    scene: SceneAttributes
    resources: ResourceSnapshot
    decision: ModuleDecision
    benchmark: BenchmarkRecord | None
