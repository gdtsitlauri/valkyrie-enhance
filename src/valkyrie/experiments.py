from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ValkyrieConfig
from .engine import ValkyrieEngine
from .metrics import compute_lpips_proxy, compute_psnr, compute_ssim


@dataclass(slots=True)
class ScenarioResult:
    seed: int
    scenario: str
    fps: float
    latency_ms: float
    psnr: float
    ssim: float
    lpips: float
    output_shape: tuple[int, int, int]


def build_synthetic_frame(seed: int, mode: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = 72, 128
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    base = np.stack(
        [
            np.repeat(x * 255, height, axis=0),
            np.repeat(y * 255, width, axis=1),
            np.repeat((1.0 - x) * 255, height, axis=0),
        ],
        axis=2,
    ).astype(np.uint8)
    if mode == "game":
        grid = ((np.indices((height, width)).sum(axis=0) % 16) < 8).astype(np.uint8) * 70
        base[:, :, 1] = np.clip(base[:, :, 1] + grid, 0, 255)
    elif mode == "film":
        noise = rng.normal(0, 18, size=base.shape).astype(np.float32)
        base = np.clip(base.astype(np.float32) * 0.75 + noise, 0, 255).astype(np.uint8)
    elif mode == "lowlight":
        base = np.clip(base.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    elif mode == "action":
        base = cv2.GaussianBlur(base, (0, 0), 1.8)
        base = np.roll(base, shift=6, axis=1)
    return base


def _degrade_reference(frame: np.ndarray, scenario: str) -> np.ndarray:
    degraded = frame.copy()
    if scenario in {"film", "action"}:
        degraded = cv2.GaussianBlur(degraded, (0, 0), 1.2)
    if scenario in {"film", "lowlight"}:
        noise = np.random.default_rng(42).normal(0, 10, degraded.shape)
        degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return degraded


def run_synthetic_benchmarks(config: ValkyrieConfig, seeds: tuple[int, ...] = (42, 43, 44)) -> list[ScenarioResult]:
    scenarios = ("game", "film", "lowlight", "action")
    rows: list[ScenarioResult] = []
    for seed in seeds:
        engine = ValkyrieEngine(config)
        for scenario in scenarios:
            source = build_synthetic_frame(seed, scenario)
            degraded = _degrade_reference(source, scenario)
            result = engine.process(degraded)
            bench = result.benchmark
            resized = cv2.resize(result.frame, (source.shape[1], source.shape[0]))
            rows.append(
                ScenarioResult(
                    seed=seed,
                    scenario=scenario,
                    fps=bench.fps if bench else 0.0,
                    latency_ms=bench.latency_ms if bench else 0.0,
                    psnr=compute_psnr(source, resized),
                    ssim=compute_ssim(source, resized),
                    lpips=compute_lpips_proxy(source, resized),
                    output_shape=tuple(int(v) for v in result.frame.shape),
                )
            )
    return rows


def run_module_ablation(base_config: ValkyrieConfig) -> dict[str, float]:
    module_names = [
        "restoration",
        "temporal_consistency",
        "latency_optimization",
        "hdr_reconstruction",
        "face_object_enhancement",
        "upscaling",
    ]
    baseline = float(np.mean([row.psnr for row in run_synthetic_benchmarks(base_config, seeds=(42,))]))
    contributions: dict[str, float] = {"full_pipeline_psnr": baseline}
    for module_name in module_names:
        config = ValkyrieConfig()
        config.runtime = base_config.runtime
        config.quality = base_config.quality
        config.learning = base_config.learning
        config.modules = type(base_config.modules)(**asdict(base_config.modules))
        setattr(config.modules, module_name, False)
        score = float(np.mean([row.psnr for row in run_synthetic_benchmarks(config, seeds=(42,))]))
        contributions[module_name] = baseline - score
    return contributions


def export_benchmark_rows(rows: list[ScenarioResult], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "seed": row.seed,
            "scenario": row.scenario,
            "fps": row.fps,
            "latency_ms": row.latency_ms,
            "psnr": row.psnr,
            "ssim": row.ssim,
            "lpips": row.lpips,
            "output_shape": list(row.output_shape),
        }
        for row in rows
    ]
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
