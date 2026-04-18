from __future__ import annotations

import cv2
import numpy as np

from ..config import ValkyrieConfig


class FaceObjectEnhancementModule:
    name = "face_object_enhancement"

    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def _enhance_roi(self, roi: np.ndarray, strength: float) -> np.ndarray:
        """Multi-scale sharpening + local contrast on a region of interest."""
        # Unsharp mask with two radii for detail recovery
        blur1 = cv2.GaussianBlur(roi, (0, 0), 0.8)
        blur2 = cv2.GaussianBlur(roi, (0, 0), 2.0)
        detail_fine = roi.astype(np.float32) - blur1.astype(np.float32)
        detail_coarse = roi.astype(np.float32) - blur2.astype(np.float32)
        sharpened = roi.astype(np.float32) + detail_fine * (strength * 0.6) + detail_coarse * (strength * 0.3)

        # Local contrast via LAB L-channel CLAHE
        sharpened_u8 = np.clip(sharpened, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(sharpened_u8, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5 + strength, tileGridSize=(4, 4))
        enhanced = cv2.merge((clahe.apply(l), a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _soft_paste(self, result: np.ndarray, roi_enhanced: np.ndarray,
                    x: int, y: int, w: int, h: int) -> None:
        """Paste enhanced ROI with feathered edges to avoid hard boundaries."""
        mask = np.ones((h, w), dtype=np.float32)
        border = max(4, min(w, h) // 8)
        for i in range(border):
            alpha = i / border
            mask[i, :] *= alpha
            mask[-(i + 1), :] *= alpha
            mask[:, i] *= alpha
            mask[:, -(i + 1)] *= alpha
        mask3 = mask[..., np.newaxis]
        orig = result[y:y + h, x:x + w].astype(np.float32)
        blended = orig * (1.0 - mask3) + roi_enhanced.astype(np.float32) * mask3
        result[y:y + h, x:x + w] = np.clip(blended, 0, 255).astype(np.uint8)

    def process(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        strength = float(np.clip(self.config.quality.face_boost, 0.0, 1.5))
        if strength < 0.01:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24)
        )
        for x, y, w, h in faces:
            # Expand ROI slightly for context
            pad = int(min(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            roi = result[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            enhanced_roi = self._enhance_roi(roi, strength)
            self._soft_paste(result, enhanced_roi, x1, y1, x2 - x1, y2 - y1)

        return result
