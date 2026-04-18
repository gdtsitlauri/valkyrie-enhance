from __future__ import annotations

import cv2
import numpy as np

from ..config import ValkyrieConfig
from ..native import NativeAccelerationManager


class TemporalConsistencyModule:
    name = "temporal_consistency"

    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config
        self.previous_frame: np.ndarray | None = None
        self.previous_gray: np.ndarray | None = None
        self.native = NativeAccelerationManager(config)

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.previous_frame is None or self.previous_frame.shape != frame.shape:
            self.previous_frame = frame.copy()
            self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame

        alpha = float(np.clip(self.config.quality.temporal_strength, 0.0, 0.5))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            # Real optical flow — Farneback dense flow
            flow = cv2.calcOpticalFlowFarneback(
                self.previous_gray, gray,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )
            # Warp previous frame along flow to align with current
            h, w = frame.shape[:2]
            map_x = (np.tile(np.arange(w), (h, 1)) + flow[..., 0]).astype(np.float32)
            map_y = (np.tile(np.arange(h), (w, 1)).T + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(self.previous_frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # Motion magnitude: reduce blending where motion is large (fast action)
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_weight = np.clip(1.0 - magnitude / 20.0, 0.0, 1.0)[..., np.newaxis].astype(np.float32)
            effective_alpha = alpha * motion_weight

            stabilized = (frame.astype(np.float32) * (1.0 - effective_alpha) +
                          warped.astype(np.float32) * effective_alpha)
            stabilized = np.clip(stabilized, 0, 255).astype(np.uint8)
        except Exception:
            # Fallback to simple blend
            stabilized = self.native.temporal_blend(frame, self.previous_frame, alpha)

        self.previous_frame = stabilized.copy()
        self.previous_gray = gray
        return stabilized
