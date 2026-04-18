from pathlib import Path

import cv2
import numpy as np

from valkyrie.engine import ValkyrieEngine
from valkyrie.ffmpeg_pipeline import ffmpeg_stream_process, probe_video_metadata
from valkyrie.native import NativeAccelerationManager, native_source_manifest
from valkyrie.real_benchmarks import benchmark_media_file
from valkyrie.reshade import generate_reshade_bundle


def _write_demo_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 32))
    assert writer.isOpened()
    for index in range(4):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 20 + index * 10
        frame[:, :, 1] = 40 + index * 10
        frame[:, :, 2] = 60 + index * 10
        writer.write(frame)
    writer.release()


def test_native_manifest_points_to_sources() -> None:
    manifest = native_source_manifest()
    assert Path(manifest["cpp"]).exists()
    assert Path(manifest["cuda"]).exists()


def test_reshade_bundle_generation(tmp_path) -> None:
    bundle = generate_reshade_bundle(str(tmp_path / "reshade"))
    assert Path(bundle["addon"]).exists()
    assert Path(bundle["ini"]).exists()


def test_real_media_benchmark_runs(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    _write_demo_video(video_path)
    rows = benchmark_media_file(ValkyrieEngine(), str(video_path), max_frames=2)
    assert any(row["system"] == "valkyrie" for row in rows)


def test_ffmpeg_stream_process_runs(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_demo_video(video_path)
    metadata = probe_video_metadata(video_path)
    summary = ffmpeg_stream_process(ValkyrieEngine(), video_path, output_path, metadata=metadata, max_frames=2)
    assert summary.frames_processed == 2
    assert output_path.exists()
