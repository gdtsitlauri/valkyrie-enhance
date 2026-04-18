from __future__ import annotations

import cv2
import numpy as np

from ..config import ValkyrieConfig


class LatencyOptimizationModule:
    name = "latency_optimization"

    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config
        self.previous_frame: np.ndarray | None = None
        self.previous_gray: np.ndarray | None = None

    def interpolate(self, frame_a: np.ndarray, frame_b: np.ndarray, t: float = 0.5) -> np.ndarray:
        """Optical flow-based frame interpolation between frame_a and frame_b at time t."""
        if frame_a.shape != frame_b.shape:
            return frame_b
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        try:
            flow_ab = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None,
                pyr_scale=0.5, levels=2, winsize=11,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            h, w = frame_a.shape[:2]
            grid_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
            grid_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)

            map_x_fwd = grid_x + flow_ab[..., 0] * t
            map_y_fwd = grid_y + flow_ab[..., 1] * t
            warped_a = cv2.remap(frame_a, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            map_x_bwd = grid_x - flow_ab[..., 0] * (1.0 - t)
            map_y_bwd = grid_y - flow_ab[..., 1] * (1.0 - t)
            warped_b = cv2.remap(frame_b, map_x_bwd, map_y_bwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            return cv2.addWeighted(warped_a, 1.0 - t, warped_b, t, 0)
        except Exception:
            return cv2.addWeighted(frame_a, 1.0 - t, frame_b, t, 0)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return the current frame, storing state for interpolation queries."""
        if self.previous_frame is None or self.previous_frame.shape != frame.shape:
            self.previous_frame = frame.copy()
            return frame
        # For single-frame path: predict by extrapolating motion slightly
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.previous_gray is None:
                self.previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                self.previous_gray, gray, None,
                pyr_scale=0.5, levels=2, winsize=11,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            # Slightly motion-compensated output (reduces perceived latency)
            h, w = frame.shape[:2]
            map_x = (np.tile(np.arange(w), (h, 1)) + flow[..., 0] * 0.1).astype(np.float32)
            map_y = (np.tile(np.arange(h), (w, 1)).T + flow[..., 1] * 0.1).astype(np.float32)
            result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            self.previous_gray = gray
        except Exception:
            result = frame
        self.previous_frame = frame.copy()
        return result
