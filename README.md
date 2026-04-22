# VALKYRIE

**Visual and Audio Luminance Enhancement via Knowledge-driven Yield and Real-time Intelligence Engine**

**Author:** George David Tsitlauri  
**Contact:** gdtsitlauri@gmail.com  
**Website:** gdtsitlauri.dev  
**GitHub:** github.com/gdtsitlauri  
**Year:** 2026

VALKYRIE is a research-oriented media-enhancement framework for real-time video and audio processing on budget NVIDIA hardware. The repository combines a modular Python pipeline, optional native CUDA kernels, benchmarking utilities, and TensorRT optimization hooks.

## Evidence Status

| Item | Current status |
| --- | --- |
| Real-time throughput benchmarks on GTX 1650 | Present |
| Quality-benchmark artifacts | Present |
| TensorRT optimization module | Present |
| Standardized full-reference super-resolution benchmark | Not fully established in the current committed snapshot |
| Production deployment evidence | Not present |

## Research Positioning

The strongest claims supported by the committed artifacts are:

> VALKYRIE demonstrates real implementation depth, a credible GTX 1650 real-time pipeline, and optional TensorRT-oriented deployment work. The current quality-benchmark snapshot is useful as an internal diagnostic, but it should not be marketed as a definitive full-reference super-resolution leaderboard.

## Current Results

### Real-time Mode

Source: `results/fps_benchmarks/summary.md`

| Scenario | Mean FPS | Best FPS | Mean Latency (ms) |
| --- | ---: | ---: | ---: |
| Action | `78.25` | `79.25` | `12.78` |
| Film | `62.84` | `64.13` | `15.92` |
| Low-light | `45.04` | `63.35` | `49.37` |
| Game | `39.14` | `57.19` | `107.04` |

These runs support the real engineering claim in the repo: some scene classes can exceed 60 FPS on a GTX 1650 in the lightweight real-time path.

### Quality Snapshot

Source: `results/quality_benchmarks/media_benchmark_summary.md`

The committed media-quality snapshot reports:

- `VALKYRIE`: mean PSNR `20.78`, mean SSIM `0.9282`, mean LPIPS `0.0575`
- 12 evaluated frames
- mean throughput `62.15 FPS` for the tested media benchmark path

Important note:

- the current committed snapshot uses a proxy reference workflow documented in `results/quality_benchmarks/README.md`
- those numbers are useful for repository-internal regression testing and model-comparison diagnostics
- they are not a substitute for a fully standardized LR$\rightarrow$HR benchmark with canonical references

## Core Modules

| Module | Purpose |
| --- | --- |
| `engine.py` | VALKYRIE coordination logic |
| `modules/` | enhancement components |
| `real_benchmarks.py` | media benchmark harness |
| `reporting.py` | summary/markdown export |
| `tensorrt_optimize.py` | TensorRT conversion and speed benchmarking |
| `native/` | optional CUDA extension path |

## TensorRT and Deployment Notes

TensorRT support is implemented in `src/valkyrie/tensorrt_optimize.py` and exposed through the optional dependency group in `pyproject.toml`. This is valid evidence of deployment-oriented optimization work, but it should still be described as prototype integration rather than mature production serving.

## Repository Layout

```text
src/valkyrie/
  modules/
  engine.py
  real_benchmarks.py
  reporting.py
  tensorrt_optimize.py
native/
configs/
paper/
results/
  fps_benchmarks/
  quality_benchmarks/
  ablation/
tests/
```

## Reproducibility

Install:

```bash
pip install -e .[dev,metrics,audio]
```

Optional TensorRT extras:

```bash
pip install -e .[tensorrt]
```

Run:

```bash
pytest
python -m valkyrie benchmark --output-dir results
python -m valkyrie benchmark-media --input video.mp4 --output-dir results
```

## Limitations

- The strongest evidence is on throughput, not on standardized perceptual quality leadership.
- Quality reporting needs a cleaner fully standardized reference protocol for headline comparison.
- The repository does not yet include deployment traces from a live application stack.

## Future Work

- Regenerate media-quality benchmarks with a strict canonical reference protocol.
- Add scene-stratified TensorRT vs PyTorch export tables.
- Extend evaluation to more varied real-world input sets and artifact classes.

## Citation

```bibtex
@misc{tsitlauri2026valkyrie,
  author = {George David Tsitlauri},
  title  = {VALKYRIE: Visual and Audio Luminance Enhancement via Knowledge-driven Yield and Real-time Intelligence Engine},
  year   = {2026},
  url    = {https://github.com/gdtsitlauri}
}
```
