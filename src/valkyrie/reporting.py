from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .types import BenchmarkRecord
from .experiments import ScenarioResult


def write_benchmark_report(record: BenchmarkRecord, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")


def write_scenario_summary(rows: list[ScenarioResult], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = sorted({row.scenario for row in rows})
    lines = [
        "# Synthetic Benchmark Summary",
        "",
        "| Scenario | Mean FPS | Mean Latency (ms) | Mean PSNR | Mean SSIM | Mean LPIPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in scenarios:
        items = [row for row in rows if row.scenario == scenario]
        lines.append(
            "| "
            + scenario
            + f" | {mean(row.fps for row in items):.2f}"
            + f" | {mean(row.latency_ms for row in items):.2f}"
            + f" | {mean(row.psnr for row in items):.2f}"
            + f" | {mean(row.ssim for row in items):.4f}"
            + f" | {mean(row.lpips for row in items):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fps_summary(rows: list[ScenarioResult], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = sorted({row.scenario for row in rows})
    lines = [
        "# FPS Benchmark Summary",
        "",
        "| Scenario | Mean FPS | Best FPS | Mean Latency (ms) | Worst Latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario in scenarios:
        items = [row for row in rows if row.scenario == scenario]
        lines.append(
            "| "
            + scenario
            + f" | {mean(row.fps for row in items):.2f}"
            + f" | {max(row.fps for row in items):.2f}"
            + f" | {mean(row.latency_ms for row in items):.2f}"
            + f" | {max(row.latency_ms for row in items):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_media_benchmark_summary(rows: list[dict[str, object]], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Media Benchmark Summary",
        "",
        "| System | Frames | Mean FPS | Mean Latency (ms) | Mean PSNR | Mean SSIM | Mean LPIPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['frames']} | {row['mean_fps']:.2f} | {row['mean_latency_ms']:.2f} | {row['mean_psnr']:.2f} | {row['mean_ssim']:.4f} | {row['mean_lpips']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
