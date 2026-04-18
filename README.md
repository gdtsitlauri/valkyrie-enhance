# VALKYRIE

**Visual and Audio Luminance Enhancement via Knowledge-driven Yield and Real-time Intelligence Engine**

An open source AI research framework for real-time, content-aware video and audio enhancement on budget GPUs. VALKYRIE introduces **VALKYRIE-APEX**, a perceptual coordination engine that intelligently activates and scales 11 enhancement modules based on scene content, under strict GPU memory and latency budgets.

**Author:** George David Tsitlauri  
**License:** MIT  
**Target hardware:** NVIDIA GTX 1650 (4 GB VRAM, CUDA 12.x, Windows 11)

---

## Results (GTX 1650, CUDA 12.4)

### Real-time Mode (ESRGANLite architecture, 2× upscale)

| Scenario  | Mean FPS | Best FPS | Mean Latency (ms) |
|-----------|----------|----------|-------------------|
| Action    | **78.25** | 79.25   | 12.78             |
| Film      | **62.84** | 64.13   | 15.92             |
| Low-light | 45.04    | 63.35    | 49.37             |
| Game      | 39.14    | 57.19    | 107.04            |

### Quality Mode (Real-ESRGAN pretrained, 720p real video)

| System            | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|-------------------|-------------|--------|---------|
| **VALKYRIE**      | **42.93**   | **0.908** | **0.00193** |
| Lanczos-4         | 76.56*      | 0.9997 | 0.000207 |
| Sharpened-linear  | 71.60*      | 0.9993 | 0.000331 |

*Classical methods compared using standard SR protocol (LR→HR vs original HR).

---

## Modules

1. **Upscaling** — ESRGANLite (real-time) + Real-ESRGAN (quality)
2. **Video Restoration** — denoising, artifact removal, sharpening
3. **Temporal Consistency** — Farneback optical flow, motion-adaptive blending
4. **Adaptive Quality** — real-time GPU load monitoring, per-module throttling
5. **Latency Optimization** — frame interpolation, motion-compensated prediction
6. **HDR Reconstruction** — ACES filmic tonemapping, BT.709→BT.2020 expansion
7. **Face & Object Enhancement** — multi-scale unsharp mask, CLAHE, feathered ROI
8. **Audio Enhancement** — STFT spectral subtraction, overlap-add reconstruction
9. **Scene Detection** — content classification, per-scene optimal settings
10. **Benchmark Suite** — PSNR, SSIM, LPIPS proxy, FPS, latency reporting
11. **Perceptual AI Engine (APEX)** — central coordinator, <1 ms overhead

---

## Quick Start

```bash
pip install -e .[dev,metrics,audio]
pytest
python -m valkyrie demo
python -m valkyrie benchmark --output-dir results
python -m valkyrie benchmark-media --input video.mp4 --output-dir results
```

### Process a video (quality mode)
```bash
python -m valkyrie process-video --input input.mp4 --output output.mp4 --max-frames 60
```

### Process a video (real-time mode)
```bash
python -m valkyrie process-video --input input.mp4 --output output.mp4 --config configs/realtime.yaml --max-frames 60
```

### Build native CUDA extension (Windows, VS 2022 + CUDA 12.4)
```bash
cd native
set DISTUTILS_USE_SDK=1
python build_native.py build_ext --inplace
```

---

## Repository Layout

```
src/valkyrie/          # Core Python package
  modules/             # 11 enhancement modules
  engine.py            # VALKYRIE-APEX coordinator
  metrics.py           # PSNR, SSIM, LPIPS proxy
  real_benchmarks.py   # Real-media SR benchmark harness
  experiments.py       # Synthetic benchmarks & ablation
  monitoring.py        # GPU/CPU resource monitoring
native/                # CUDA C++ kernels
  valkyrie_native.cpp  # PyTorch extension bindings
  valkyrie_native_kernel.cu  # Bilinear, bilateral, ACES, temporal kernels
configs/               # YAML configs
  realtime.yaml        # ESRGANLite real-time mode
  research_baseline.yaml
paper/                 # IEEE paper
  valkyrie_paper.tex
results/               # Benchmark artifacts
  quality_benchmarks/
  fps_benchmarks/
  ablation/
integrations/reshade/  # Windows ReShade scaffolding
tests/                 # Pytest test suite
```

---

## Novel Contributions

- **VALKYRIE-APEX**: first content-aware perceptual AI engine for coordinating multi-module real-time video enhancement
- **Unified framework**: video + audio + gaming enhancement in one open source pipeline
- **Budget GPU focus**: designed for GTX 1650, excluded from DLSS/FSR
- **Dual-mode upscaling**: real-time ESRGANLite + quality Real-ESRGAN in one codebase

---

## License

MIT License — Copyright (c) 2026 George David Tsitlauri
