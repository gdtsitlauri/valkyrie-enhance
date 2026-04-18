from __future__ import annotations

import numpy as np

from .config import ValkyrieConfig
from .monitoring import ResourceMonitor
from .native import NativeAccelerationManager
from .modules import (
    AdaptiveQualityModule,
    AudioEnhancementModule,
    BenchmarkSuiteModule,
    FaceObjectEnhancementModule,
    HDRReconstructionModule,
    LatencyOptimizationModule,
    PerceptualAIEngineModule,
    SceneDetectionModule,
    TemporalConsistencyModule,
    UpscalingModule,
    VideoRestorationModule,
)
from .types import PipelineResult


class ValkyrieEngine:
    """Central VALKYRIE-APEX orchestration pipeline."""

    def __init__(self, config: ValkyrieConfig | None = None) -> None:
        self.config = config or ValkyrieConfig()
        self.monitor = ResourceMonitor(self.config)
        self.native = NativeAccelerationManager(self.config)
        self.scene = SceneDetectionModule()
        self.adaptive = AdaptiveQualityModule(self.config)
        self.perceptual = PerceptualAIEngineModule(self.config)
        self.upscaling = UpscalingModule(self.config)
        self.restoration = VideoRestorationModule(self.config)
        self.temporal = TemporalConsistencyModule(self.config)
        self.latency = LatencyOptimizationModule(self.config)
        self.hdr = HDRReconstructionModule(self.config)
        self.face_object = FaceObjectEnhancementModule(self.config)
        self.audio = AudioEnhancementModule()
        self.benchmark = BenchmarkSuiteModule()

    def process(self, frame: np.ndarray, audio: np.ndarray | None = None) -> PipelineResult:
        started = self.benchmark.begin()
        resources = self.monitor.snapshot()
        scene = self.scene.analyze(frame)
        decision = self.perceptual.refine(self.adaptive.decide(scene, resources))

        output = frame.copy()
        if decision.enabled.get("restoration", False) and self.config.modules.restoration:
            output = self.restoration.process(output, artifact_level=scene.artifact_level)
        if decision.enabled.get("temporal_consistency", False) and self.config.modules.temporal_consistency:
            output = self.temporal.process(output)
        if decision.enabled.get("latency_optimization", False) and self.config.modules.latency_optimization:
            output = self.latency.process(output)
        if decision.enabled.get("hdr_reconstruction", False) and self.config.modules.hdr_reconstruction:
            output = self.hdr.process(output)
        if decision.enabled.get("face_object_enhancement", False) and self.config.modules.face_object_enhancement:
            output = self.face_object.process(output)
        if decision.enabled.get("upscaling", False) and self.config.modules.upscaling:
            scale = self.config.quality.scale_factor * decision.quality_multipliers.get("upscaling", 1.0)
            output = self.upscaling.process(output, scale_factor=max(1.0, scale))

        enhanced_audio = self.audio.process(audio) if self.config.modules.audio_enhancement else audio
        benchmark = self.benchmark.end(started, output, resources) if self.config.modules.benchmark_suite else None
        return PipelineResult(
            frame=output,
            audio=enhanced_audio,
            scene=scene,
            resources=resources,
            decision=decision,
            benchmark=benchmark,
        )
