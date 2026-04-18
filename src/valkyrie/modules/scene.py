from __future__ import annotations

import cv2
import numpy as np

from ..types import SceneAttributes


class SceneDetectionModule:
    name = "scene_detection"

    def __init__(self) -> None:
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def analyze(self, frame: np.ndarray) -> SceneAttributes:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        detail = float(np.var(laplacian))
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.mean(edges > 0))
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        content_type = "game" if edge_density > 0.18 else "film" if brightness < 0.45 else "real_video"
        motion_level = "high" if detail > 450 else "medium" if detail > 120 else "static"
        lighting = "night" if brightness < 0.3 else "day" if brightness > 0.7 else "balanced"
        artifact_level = float(np.clip(1.0 - detail / 800.0, 0.0, 1.0))
        return SceneAttributes(
            content_type=content_type,
            motion_level=motion_level,
            lighting=lighting,
            has_faces=len(faces) > 0,
            artifact_level=artifact_level,
            style="cinematic" if brightness < 0.4 else "clean",
        )
