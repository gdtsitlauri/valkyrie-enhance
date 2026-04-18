from __future__ import annotations

import cv2
import numpy as np

from ..config import ValkyrieConfig


# ACES filmic tonemapping coefficients
_ACES_A = 2.51
_ACES_B = 0.03
_ACES_C = 2.43
_ACES_D = 0.59
_ACES_E = 0.14

# BT.709 → BT.2020 color matrix (linear RGB)
_BT709_TO_BT2020 = np.array([
    [0.6274, 0.3293, 0.0433],
    [0.0691, 0.9195, 0.0114],
    [0.0164, 0.0880, 0.8956],
], dtype=np.float32)


def _aces_tonemap(x: np.ndarray) -> np.ndarray:
    return np.clip((x * (_ACES_A * x + _ACES_B)) / (x * (_ACES_C * x + _ACES_D) + _ACES_E), 0.0, 1.0)


class HDRReconstructionModule:
    name = "hdr_reconstruction"

    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config

    def process(self, frame: np.ndarray) -> np.ndarray:
        strength = float(np.clip(self.config.quality.hdr_strength, 0.0, 1.0))
        if strength < 0.01:
            return frame

        img = frame.astype(np.float32) / 255.0

        # 1. Linearize (approximate inverse sRGB gamma)
        linear = np.power(np.clip(img, 0, 1), 2.2)

        # 2. Dynamic range expansion: lift shadows, compress highlights
        exposure = 1.0 + strength * 0.5
        linear_exposed = linear * exposure

        # 3. ACES filmic tonemapping
        tonemapped = _aces_tonemap(linear_exposed)

        # 4. Optional gamut expansion BT.709 → BT.2020 (partial, perceptual)
        if strength > 0.3:
            gamut_strength = (strength - 0.3) / 0.7
            b, g, r = tonemapped[..., 0], tonemapped[..., 1], tonemapped[..., 2]
            rgb = np.stack([r, g, b], axis=-1)
            rgb_2020 = rgb @ _BT709_TO_BT2020.T
            rgb_2020 = np.clip(rgb_2020, 0, 1)
            rgb_blended = rgb * (1.0 - gamut_strength * 0.3) + rgb_2020 * (gamut_strength * 0.3)
            tonemapped = np.stack([rgb_blended[..., 2], rgb_blended[..., 1], rgb_blended[..., 0]], axis=-1)

        # 5. Re-encode to sRGB gamma
        srgb = np.power(np.clip(tonemapped, 0, 1), 1.0 / 2.2)

        # 6. Local contrast enhancement via CLAHE in L channel
        result_uint8 = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(result_uint8, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5 + strength, tileGridSize=(8, 8))
        lab_enhanced = cv2.merge((clahe.apply(l), a, b_ch))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Blend with original based on strength
        out = cv2.addWeighted(frame, 1.0 - strength * 0.6, enhanced, strength * 0.6, 0)
        return out.astype(np.uint8)
