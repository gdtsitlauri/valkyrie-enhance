from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RuntimeConfig:
    target_fps: int = 60
    vram_budget_gb: float = 3.5
    prefer_cuda: bool = True
    enable_cpu_fallback: bool = True
    enable_native_acceleration: bool = True
    enable_ffmpeg_pipeline: bool = True
    live_low_latency_mode: bool = True
    reshade_integration: bool = False


@dataclass(slots=True)
class QualityConfig:
    scale_factor: float = 2.0
    denoise_strength: float = 0.35
    sharpen_strength: float = 0.2
    temporal_strength: float = 0.45
    hdr_strength: float = 0.3
    face_boost: float = 0.25
    interpolation_factor: int = 2
    use_realesrgan: bool = True


@dataclass(slots=True)
class ModuleToggleConfig:
    upscaling: bool = True
    restoration: bool = True
    temporal_consistency: bool = True
    adaptive_quality: bool = True
    latency_optimization: bool = True
    hdr_reconstruction: bool = True
    face_object_enhancement: bool = True
    audio_enhancement: bool = True
    scene_detection: bool = True
    benchmark_suite: bool = True
    perceptual_ai_engine: bool = True


@dataclass(slots=True)
class LearningConfig:
    preference_bias: dict[str, float] = field(
        default_factory=lambda: {
            "sharpness": 0.5,
            "stability": 0.5,
            "latency": 0.5,
            "hdr": 0.5,
        }
    )
    adaptation_rate: float = 0.1


@dataclass(slots=True)
class ValkyrieConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    modules: ModuleToggleConfig = field(default_factory=ModuleToggleConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def config_from_dict(data: dict[str, Any]) -> ValkyrieConfig:
    config = ValkyrieConfig()
    return _merge_dataclass(config, data)


def load_config(path: str | Path) -> ValkyrieConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration file must contain a mapping at the top level")
    return config_from_dict(payload)


def dump_config(config: ValkyrieConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
