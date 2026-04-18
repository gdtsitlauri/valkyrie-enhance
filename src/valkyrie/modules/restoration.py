from __future__ import annotations

import cv2
import numpy as np

from ..config import ValkyrieConfig
from ..native import NativeAccelerationManager


class VideoRestorationModule:
    name = "restoration"

    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config
        self.native = NativeAccelerationManager(config)

    def process(self, frame: np.ndarray, artifact_level: float = 0.2) -> np.ndarray:
        restored = self.native.restore(
            frame,
            denoise_strength=self.config.quality.denoise_strength + artifact_level,
            sharpen_strength=self.config.quality.sharpen_strength,
        )
        sigma = 1.0 + artifact_level
        blurred = cv2.GaussianBlur(restored, (0, 0), sigmaX=sigma)
        refined = cv2.addWeighted(restored, 1.0 + self.config.quality.sharpen_strength * 0.5, blurred, -self.config.quality.sharpen_strength * 0.5, 0)
        return np.clip(refined, 0, 255).astype(np.uint8)
