from __future__ import annotations

from ..config import ValkyrieConfig
from ..types import ModuleDecision, ResourceSnapshot, SceneAttributes


class AdaptiveQualityModule:
    name = "adaptive_quality"

    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config

    def decide(self, scene: SceneAttributes, resources: ResourceSnapshot) -> ModuleDecision:
        overload = resources.gpu_load > 0.9 or resources.vram_used_gb > self.config.runtime.vram_budget_gb
        degraded = resources.fps_estimate < self.config.runtime.target_fps * 0.9
        quality = 0.6 if overload else 0.85 if degraded else 1.0
        enabled = {
            "hdr_reconstruction": not overload and scene.lighting != "balanced",
            "face_object_enhancement": scene.has_faces or scene.content_type in {"film", "real_video"},
            "latency_optimization": scene.motion_level == "high",
            "restoration": scene.artifact_level > 0.15,
            "temporal_consistency": scene.motion_level != "static",
            "upscaling": True,
            "audio_enhancement": True,
        }
        if overload and resources.emergency:
            enabled["hdr_reconstruction"] = False
            enabled["face_object_enhancement"] = False
        return ModuleDecision(
            enabled=enabled,
            quality_multipliers={name: quality for name in enabled},
            rationale={
                "overload": overload,
                "degraded": degraded,
                "scene_type": scene.content_type,
            },
        )
