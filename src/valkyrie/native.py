from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ValkyrieConfig


@dataclass(slots=True)
class NativeStatus:
    available: bool
    backend: str
    reason: str


class NativeAccelerationManager:
    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config
        self.module: Any | None = None
        self.status = self._load()

    def _load(self) -> NativeStatus:
        if not self.config.runtime.enable_native_acceleration:
            return NativeStatus(False, "disabled", "native acceleration disabled by config")
        try:
            self.module = importlib.import_module("valkyrie_native")
            return NativeStatus(True, "compiled_extension", "loaded compiled valkyrie_native module")
        except Exception as exc:
            return NativeStatus(False, "opencv_fallback", str(exc))

    def describe(self) -> dict[str, str | bool]:
        return {
            "available": self.status.available,
            "backend": self.status.backend,
            "reason": self.status.reason,
        }

    def upsample(self, frame: np.ndarray, scale_factor: float, sharpen_strength: float) -> np.ndarray:
        if self.module is not None and hasattr(self.module, "upsample_rgb"):
            return self.module.upsample_rgb(frame, scale_factor, sharpen_strength)
        height, width = frame.shape[:2]
        resized = cv2.resize(
            frame,
            (max(1, int(width * scale_factor)), max(1, int(height * scale_factor))),
            interpolation=cv2.INTER_LINEAR,
        )
        if sharpen_strength <= 0:
            return resized
        blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
        sharpened = cv2.addWeighted(resized, 1.0 + sharpen_strength, blurred, -sharpen_strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def temporal_blend(self, current: np.ndarray, previous: np.ndarray, alpha: float) -> np.ndarray:
        if self.module is not None and hasattr(self.module, "temporal_blend_rgb"):
            return self.module.temporal_blend_rgb(current, previous, alpha)
        return cv2.addWeighted(current, 1.0 - alpha, previous, alpha, 0)

    def restore(self, frame: np.ndarray, denoise_strength: float, sharpen_strength: float) -> np.ndarray:
        if self.module is not None and hasattr(self.module, "restore_rgb"):
            return self.module.restore_rgb(frame, denoise_strength, sharpen_strength)
        denoise_h = max(3, int(10 * denoise_strength))
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, denoise_h, denoise_h, 7, 21)
        blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
        restored = cv2.addWeighted(denoised, 1.0 + sharpen_strength, blurred, -sharpen_strength, 0)
        return np.clip(restored, 0, 255).astype(np.uint8)


def native_source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2] / "native"
    return {
        "cpp": str(root / "valkyrie_native.cpp"),
        "cuda": str(root / "valkyrie_native_kernel.cu"),
        "build_script": str(root / "build_native.py"),
    }
