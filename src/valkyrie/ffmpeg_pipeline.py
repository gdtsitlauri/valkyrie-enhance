from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import ValkyrieEngine


@dataclass(slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float


@dataclass(slots=True)
class FFmpegPipelineSummary:
    frames_processed: int
    input_path: str
    output_path: str
    width: int
    height: int
    fps: float


def probe_video_metadata(path: str | Path) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    rate_parts = stream["r_frame_rate"].split("/")
    fps = float(rate_parts[0]) / max(float(rate_parts[1]), 1.0)
    return VideoMetadata(width=int(stream["width"]), height=int(stream["height"]), fps=fps)


def ffmpeg_stream_process(
    engine: ValkyrieEngine,
    input_path: str | Path,
    output_path: str | Path,
    metadata: VideoMetadata,
    max_frames: int | None = None,
) -> FFmpegPipelineSummary:
    frame_size = metadata.width * metadata.height * 3
    decoder_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-an",
            "-i",
            str(input_path),
            "-vsync",
            "0",
    ]
    if max_frames is not None:
        decoder_command.extend(["-frames:v", str(max_frames)])
    decoder_command.extend(
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
    )
    decoder = subprocess.Popen(
        decoder_command,
        stdout=subprocess.PIPE,
        bufsize=frame_size * 4,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(metadata.width * engine.config.quality.scale_factor)}x{int(metadata.height * engine.config.quality.scale_factor)}",
            "-r",
            f"{metadata.fps:.4f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        bufsize=frame_size * 8,
    )
    assert decoder.stdout is not None
    assert encoder.stdin is not None
    frames_processed = 0
    while True:
        if max_frames is not None and frames_processed >= max_frames:
            break
        raw = decoder.stdout.read(frame_size)
        if len(raw) != frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(metadata.height, metadata.width, 3)
        result = engine.process(frame)
        encoder.stdin.write(result.frame.tobytes())
        frames_processed += 1
    decoder.stdout.close()
    encoder.stdin.close()
    decoder.wait()
    encoder.wait()
    return FFmpegPipelineSummary(
        frames_processed=frames_processed,
        input_path=str(input_path),
        output_path=str(output_path),
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
    )
