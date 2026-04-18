from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .config import ValkyrieConfig, dump_config, load_config
from .engine import ValkyrieEngine
from .experiments import export_benchmark_rows, run_module_ablation, run_synthetic_benchmarks
from .ffmpeg_pipeline import ffmpeg_stream_process, probe_video_metadata
from .io import process_video_file
from .real_benchmarks import benchmark_media_file
from .reporting import write_fps_summary, write_media_benchmark_summary, write_scenario_summary
from .reshade import generate_reshade_bundle


def build_demo_frame() -> np.ndarray:
    h, w = 180, 320
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    frame = np.stack(
        [
            (x * 255).repeat(h, axis=0),
            (y * 255).repeat(w, axis=1),
            ((1.0 - x) * 255).repeat(h, axis=0),
        ],
        axis=2,
    )
    return frame.astype(np.uint8)


def current_python() -> str:
    return sys.executable or "python"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VALKYRIE pipeline and research utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run a synthetic demo frame through the pipeline")
    demo_parser.add_argument("--config", type=str, help="Optional YAML config path")

    process_parser = subparsers.add_parser("process-video", help="Process a video file with VALKYRIE")
    process_parser.add_argument("--input", required=True, type=str, help="Input video path")
    process_parser.add_argument("--output", required=True, type=str, help="Output video path")
    process_parser.add_argument("--config", type=str, help="Optional YAML config path")
    process_parser.add_argument("--max-frames", type=int, help="Optional frame limit for quick runs")

    live_parser = subparsers.add_parser("live-ffmpeg", help="Process media through an FFmpeg rawvideo low-latency pipe")
    live_parser.add_argument("--input", required=True, type=str, help="Input media path")
    live_parser.add_argument("--output", required=True, type=str, help="Output media path")
    live_parser.add_argument("--config", type=str, help="Optional YAML config path")
    live_parser.add_argument("--max-frames", type=int, help="Optional frame limit for quick runs")

    bench_parser = subparsers.add_parser("benchmark", help="Run synthetic benchmark suite")
    bench_parser.add_argument("--config", type=str, help="Optional YAML config path")
    bench_parser.add_argument("--output-dir", type=str, default="results", help="Directory for benchmark artifacts")

    media_bench_parser = subparsers.add_parser("benchmark-media", help="Benchmark VALKYRIE and baselines on a real media file")
    media_bench_parser.add_argument("--input", required=True, type=str, help="Input media path")
    media_bench_parser.add_argument("--output-dir", type=str, default="results", help="Directory for benchmark artifacts")
    media_bench_parser.add_argument("--config", type=str, help="Optional YAML config path")
    media_bench_parser.add_argument("--max-frames", type=int, default=120, help="Frame limit for fast benchmark runs")

    reshade_parser = subparsers.add_parser("init-reshade", help="Generate Windows ReShade integration scaffolding")
    reshade_parser.add_argument("--output-dir", type=str, default="integrations/reshade", help="Directory for generated ReShade bundle")

    validate_parser = subparsers.add_parser("run-all", help="Run the full local VALKYRIE validation stack")
    validate_parser.add_argument("--output-dir", type=str, default="results", help="Directory for generated artifacts")
    validate_parser.add_argument("--media", type=str, help="Optional input media path for real-media and live FFmpeg validation")
    validate_parser.add_argument("--max-frames", type=int, default=24, help="Frame limit for media validation runs")

    dump_parser = subparsers.add_parser("dump-config", help="Write a default YAML config")
    dump_parser.add_argument("--output", required=True, type=str, help="Destination YAML path")

    args = parser.parse_args()
    config = load_config(args.config) if getattr(args, "config", None) else ValkyrieConfig()

    if args.command == "demo":
        engine = ValkyrieEngine(config)
        result = engine.process(build_demo_frame(), np.random.randn(2048).astype(np.float32) * 0.01)
        payload = {
            "output_shape": list(result.frame.shape),
            "scene": asdict(result.scene),
            "resources": asdict(result.resources),
            "decision": {
                "enabled": result.decision.enabled,
                "quality_multipliers": result.decision.quality_multipliers,
                "rationale": result.decision.rationale,
            },
            "benchmark": asdict(result.benchmark) if result.benchmark else None,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "process-video":
        engine = ValkyrieEngine(config)
        summary = process_video_file(engine, args.input, args.output, max_frames=args.max_frames)
        print(json.dumps(asdict(summary), indent=2))
        return

    if args.command == "live-ffmpeg":
        engine = ValkyrieEngine(config)
        metadata = probe_video_metadata(args.input)
        summary = ffmpeg_stream_process(engine, args.input, args.output, metadata=metadata, max_frames=args.max_frames)
        print(json.dumps(asdict(summary), indent=2))
        return

    if args.command == "benchmark":
        rows = run_synthetic_benchmarks(config)
        output_dir = Path(args.output_dir)
        export_benchmark_rows(rows, output_dir / "quality_benchmarks" / "synthetic_benchmark_rows.json")
        write_scenario_summary(rows, output_dir / "quality_benchmarks" / "summary.md")
        write_fps_summary(rows, output_dir / "fps_benchmarks" / "summary.md")
        ablation = run_module_ablation(config)
        (output_dir / "ablation").mkdir(parents=True, exist_ok=True)
        (output_dir / "ablation" / "module_ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
        payload = {
            "rows": len(rows),
            "summary": str(output_dir / "quality_benchmarks" / "summary.md"),
            "fps_summary": str(output_dir / "fps_benchmarks" / "summary.md"),
            "ablation": str(output_dir / "ablation" / "module_ablation.json"),
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "benchmark-media":
        engine = ValkyrieEngine(config)
        output_dir = Path(args.output_dir)
        rows = benchmark_media_file(engine, args.input, max_frames=args.max_frames)
        destination = output_dir / "quality_benchmarks" / "media_benchmark_summary.md"
        write_media_benchmark_summary(rows, destination)
        json_path = output_dir / "quality_benchmarks" / "media_benchmark_rows.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps({"rows": len(rows), "summary": str(destination), "json": str(json_path)}, indent=2))
        return

    if args.command == "init-reshade":
        bundle = generate_reshade_bundle(args.output_dir)
        print(json.dumps(bundle, indent=2))
        return

    if args.command == "run-all":
        import cv2  # noqa: PLC0415

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_media = output_dir / "local_media_input.mp4"
        if args.media:
            media_path = Path(args.media)
        else:
            # Generate test video with OpenCV — no ffmpeg required
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(generated_media), fourcc, 24.0, (160, 90))
            for i in range(48):  # 2 seconds at 24 fps
                frame = np.zeros((90, 160, 3), dtype=np.uint8)
                frame[:, :, 0] = int(i * 5) % 256  # blue channel sweep
                frame[:, :, 2] = int(255 - i * 5) % 256  # red channel sweep
                writer.write(frame)
            writer.release()
            media_path = generated_media

        _build_env = {"DISTUTILS_USE_SDK": "1", **dict(__import__("os").environ)}
        native_build = subprocess.run(
            [current_python(), "build_native.py", "build_ext", "--inplace"],
            cwd=str(Path.cwd() / "native"),
            env=_build_env,
            capture_output=True,
            text=True,
        )
        if native_build.returncode != 0:
            native_build = subprocess.run(
                [current_python(), "build_native.py", "build_ext", "--inplace"],
                cwd=str(Path.cwd() / "native"),
                env={"VALKYRIE_BUILD_CUDA": "0", "DISTUTILS_USE_SDK": "1", **dict(__import__("os").environ)},
                capture_output=True,
                text=True,
                check=True,
            )

        tests = subprocess.run(["pytest", "-q"], check=True, capture_output=True, text=True)
        subprocess.run([current_python(), "-m", "valkyrie", "benchmark", "--output-dir", str(output_dir)], check=True)
        subprocess.run(
            [current_python(), "-m", "valkyrie", "benchmark-media", "--input", str(media_path), "--output-dir", str(output_dir), "--max-frames", str(args.max_frames)],
            check=True,
        )
        # live-ffmpeg step is optional — skip gracefully if ffmpeg is not on PATH
        live_ffmpeg_result: str = "skipped (ffmpeg not found)"
        try:
            import shutil
            if shutil.which("ffmpeg"):
                subprocess.run(
                    [current_python(), "-m", "valkyrie", "live-ffmpeg", "--input", str(media_path), "--output", str(output_dir / "live_ffmpeg_output.mp4"), "--max-frames", str(min(args.max_frames, 12))],
                    check=True,
                )
                live_ffmpeg_result = "ok"
        except Exception as exc:
            live_ffmpeg_result = f"failed: {exc}"
        bundle = generate_reshade_bundle("integrations/reshade")
        payload = {
            "native_build_returncode": native_build.returncode,
            "native_build_stdout": native_build.stdout.strip(),
            "native_build_stderr": native_build.stderr.strip(),
            "tests": tests.stdout.strip(),
            "live_ffmpeg": live_ffmpeg_result,
            "output_dir": str(output_dir),
            "media_input": str(media_path),
            "reshade": bundle,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "dump-config":
        dump_config(ValkyrieConfig(), args.output)
        print(json.dumps({"written": args.output}, indent=2))
        return


if __name__ == "__main__":
    main()
