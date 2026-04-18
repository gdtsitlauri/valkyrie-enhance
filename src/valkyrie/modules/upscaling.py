from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ..config import ValkyrieConfig
from ..native import NativeAccelerationManager

# Cache dir for downloaded model weights
_WEIGHTS_DIR = Path(os.environ.get("VALKYRIE_WEIGHTS_DIR", Path.home() / ".cache" / "valkyrie" / "weights"))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * 0.2


class ESRGANLite(nn.Module):
    """Lightweight ESRGAN-inspired upscaler for real-time GTX 1650 inference."""

    def __init__(self, scale: int = 2, channels: int = 32, num_blocks: int = 6) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = feat + self.tail(self.body(feat))
        return self.upsample(feat)


def _try_load_realesrgan(scale: int, device: str):
    """Try to load a pretrained RealESRGAN upsampler. Returns None on failure."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore[import]
        from realesrgan import RealESRGANer  # type: ignore[import]

        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        if scale == 4:
            model_name = "RealESRGAN_x4plus"
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            out_scale = 4
            num_feat = 64
            num_block = 23
        else:
            model_name = "RealESRGAN_x2plus"
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            out_scale = 2
            num_feat = 64
            num_block = 23

        weights_path = _WEIGHTS_DIR / f"{model_name}.pth"
        if not weights_path.exists():
            import urllib.request
            print(f"[VALKYRIE] Downloading {model_name} weights (~65MB)...")
            urllib.request.urlretrieve(url, str(weights_path))
            print(f"[VALKYRIE] Weights saved to {weights_path}")

        rrdb_model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=num_feat, num_block=num_block,
            num_grow_ch=32, scale=out_scale,
        )
        upsampler = RealESRGANer(
            scale=out_scale,
            model_path=str(weights_path),
            model=rrdb_model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=(device == "cuda"),
            device=torch.device(device),
        )
        return upsampler
    except Exception as exc:
        print(f"[VALKYRIE] RealESRGAN unavailable ({exc}), using ESRGANLite fallback")
        return None


class UpscalingModule:
    name = "upscaling"

    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.runtime.prefer_cuda else "cpu"
        self._realesrgan_cache: dict[int, object] = {}
        self._lite_cache: dict[int, ESRGANLite] = {}
        self.native = NativeAccelerationManager(config)

    def _get_realesrgan(self, scale: int):
        if scale not in self._realesrgan_cache:
            self._realesrgan_cache[scale] = _try_load_realesrgan(scale, self.device)
        return self._realesrgan_cache[scale]

    def _get_lite(self, scale: int) -> ESRGANLite:
        if scale not in self._lite_cache:
            model = ESRGANLite(scale=scale, channels=32, num_blocks=6).to(self.device).eval()
            if self.device == "cuda":
                model = model.half()
            self._lite_cache[scale] = model
        return self._lite_cache[scale]

    def process(self, frame: np.ndarray, scale_factor: float | None = None) -> np.ndarray:
        factor = scale_factor or self.config.quality.scale_factor
        h, w = frame.shape[:2]
        target_h, target_w = int(h * factor), int(w * factor)
        scale = max(2, round(factor))

        # Try Real-ESRGAN first (pretrained, high quality) — skipped in real-time mode
        upsampler = self._get_realesrgan(scale) if self.config.quality.use_realesrgan else None
        if upsampler is not None:
            try:
                # RealESRGANer expects BGR uint8
                output, _ = upsampler.enhance(frame, outscale=factor)
                if output.shape[:2] != (target_h, target_w):
                    output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                return output
            except Exception:
                pass

        # Fallback: ESRGANLite (architecture correct, random weights)
        try:
            model = self._get_lite(scale)
            tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            if self.device == "cuda":
                tensor = tensor.half().cuda()
            with torch.inference_mode():
                out = model(tensor).squeeze(0).permute(1, 2, 0)
                if self.device == "cuda":
                    out = out.float()
                result = out.cpu().numpy()
            result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            if result.shape[:2] != (target_h, target_w):
                result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            return result
        except Exception:
            return self.native.upsample(frame, factor, self.config.quality.sharpen_strength)
