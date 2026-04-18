from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .engine import ValkyrieEngine
from .types import BenchmarkRecord


@dataclass(slots=True)
class VideoProcessingSummary:
    frames_processed: int
    input_fps: float
    output_path: str
    average_fps: float
    average_latency_ms: float


def process_video_file(
    engine: ValkyrieEngine,
    input_path: str | Path,
    output_path: str | Path,
    max_frames: int | None = None,
) -> VideoProcessingSummary:
    source = cv2.VideoCapture(str(input_path))
    if not source.isOpened():
        raise FileNotFoundError(f"unable to open video input: {input_path}")

    fps = source.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first_frame = source.read()
    if not ok:
        source.release()
        raise ValueError("video input contains no readable frames")

    first_result = engine.process(first_frame)
    height, width = first_result.frame.shape[:2]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        source.release()
        raise RuntimeError(f"unable to create output video: {output_path}")

    metrics: list[BenchmarkRecord] = []
    writer.write(first_result.frame)
    if first_result.benchmark is not None:
        metrics.append(first_result.benchmark)
    processed = 1

    while True:
        if max_frames is not None and processed >= max_frames:
            break
        ok, frame = source.read()
        if not ok:
            break
        result = engine.process(frame)
        writer.write(result.frame)
        if result.benchmark is not None:
            metrics.append(result.benchmark)
        processed += 1

    source.release()
    writer.release()
    avg_fps = float(np.mean([item.fps for item in metrics])) if metrics else 0.0
    avg_latency = float(np.mean([item.latency_ms for item in metrics])) if metrics else 0.0
    return VideoProcessingSummary(
        frames_processed=processed,
        input_fps=float(fps),
        output_path=str(destination),
        average_fps=avg_fps,
        average_latency_ms=avg_latency,
    )
