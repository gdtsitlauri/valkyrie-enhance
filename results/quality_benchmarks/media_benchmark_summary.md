# Media Benchmark Summary

This committed snapshot is a repository-internal diagnostic export.

- It is useful for regression tracking and for comparing local VALKYRIE configurations.
- It is not a canonical public super-resolution leaderboard.
- The current rows use a proxy reference workflow rather than a strict standardized LR->HR benchmark protocol.

| System | Frames | Mean FPS | Mean Latency (ms) | Mean PSNR | Mean SSIM | Mean LPIPS | Reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| valkyrie | 12 | 62.15 | 16.09 | 20.78 | 0.9282 | 0.0575 | proxy reference (bicubic-upscaled input) |
| bicubic | 12 | 6118.57 | 0.16 | 100.00 | 1.0000 | 0.0000 | proxy reference (identity to benchmark input) |
| lanczos | 12 | 986.00 | 1.01 | 39.96 | 0.9915 | 0.0020 | proxy reference (bicubic-upscaled input) |
| sharpen | 12 | 4044.02 | 0.25 | 37.80 | 0.9946 | 0.0025 | proxy reference (bicubic-upscaled input) |
