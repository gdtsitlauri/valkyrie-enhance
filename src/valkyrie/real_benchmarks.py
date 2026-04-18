from __future__ import annotations

from statistics import mean
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .engine import ValkyrieEngine
from .metrics import compute_lpips_proxy, compute_psnr, compute_ssim


def _apply_baseline(frame: np.ndarray, mode: str, scale_factor: float) -> np.ndarray:
    height, width = frame.shape[:2]
    target_size = (max(1, int(width * scale_factor)), max(1, int(height * scale_factor)))
    if mode == "bicubic":
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
    if mode == "lanczos":
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
    if mode == "sharpen":
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
        return cv2.addWeighted(resized, 1.2, blurred, -0.2, 0)
    raise ValueError(f"unknown baseline mode: {mode}")


def benchmark_media_file(engine: ValkyrieEngine, input_path: str, max_frames: int = 120) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"unable to open media benchmark input: {input_path}")
    rows: list[dict[str, Any]] = []
    frame_index = 0
    per_system: dict[str, list[dict[str, float]]] = {"valkyrie": [], "bicubic": [], "lanczos": [], "sharpen": []}
    scale = engine.config.quality.scale_factor
    while frame_index < max_frames:
        ok, hr_frame = capture.read()
        if not ok:
            break
        # Standard SR protocol: downscale to LR, upscale back, compare vs original HR
        lr_w = max(1, int(hr_frame.shape[1] / scale))
        lr_h = max(1, int(hr_frame.shape[0] / scale))
        lr_frame = cv2.resize(hr_frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        reference = hr_frame  # original full-res frame is ground truth

        started = perf_counter()
        valkyrie_result = engine.process(lr_frame).frame
        valkyrie_time = (perf_counter() - started) * 1000.0
        per_system["valkyrie"].append(
            {
                "latency_ms": valkyrie_time,
                "psnr": compute_psnr(reference, valkyrie_result),
                "ssim": compute_ssim(reference, valkyrie_result),
                "lpips": compute_lpips_proxy(reference, valkyrie_result),
            }
        )

        for mode in ("bicubic", "lanczos", "sharpen"):
            started = perf_counter()
            output = _apply_baseline(lr_frame, mode, scale)
            elapsed = (perf_counter() - started) * 1000.0
            per_system[mode].append(
                {
                    "latency_ms": elapsed,
                    "psnr": compute_psnr(reference, output),
                    "ssim": compute_ssim(reference, output),
                    "lpips": compute_lpips_proxy(reference, output),
                }
            )
        frame_index += 1
    capture.release()

    for system, metrics in per_system.items():
        if not metrics:
            continue
        mean_latency = mean(item["latency_ms"] for item in metrics)
        rows.append(
            {
                "system": system,
                "frames": len(metrics),
                "mean_latency_ms": mean_latency,
                "mean_fps": 1000.0 / mean_latency if mean_latency > 0 else 0.0,
                "mean_psnr": mean(item["psnr"] for item in metrics),
                "mean_ssim": mean(item["ssim"] for item in metrics),
                "mean_lpips": mean(item["lpips"] for item in metrics),
                "reference": "original HR frame (LR→HR SR protocol)",
            }
        )
    return rows
