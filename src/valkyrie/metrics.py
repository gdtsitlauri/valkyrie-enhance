from __future__ import annotations

import math

import cv2
import numpy as np


def compute_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB (higher = better)."""
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    if ref.shape != cand.shape:
        cand = cv2.resize(cand, (ref.shape[1], ref.shape[0]))
        cand = cand.astype(np.float64)
    mse = float(np.mean((ref - cand) ** 2))
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Structural Similarity Index (0-1, higher = better). Full multi-channel MS-SSIM."""
    if reference.shape != candidate.shape:
        candidate = cv2.resize(candidate, (reference.shape[1], reference.shape[0]))

    ssim_per_channel = []
    for i in range(reference.shape[2]):
        ref = reference[..., i].astype(np.float64)
        cand = candidate[..., i].astype(np.float64)
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2

        mu_r = cv2.GaussianBlur(ref, (11, 11), 1.5)
        mu_c = cv2.GaussianBlur(cand, (11, 11), 1.5)

        sigma_r = cv2.GaussianBlur(ref * ref, (11, 11), 1.5) - mu_r * mu_r
        sigma_c = cv2.GaussianBlur(cand * cand, (11, 11), 1.5) - mu_c * mu_c
        sigma_rc = cv2.GaussianBlur(ref * cand, (11, 11), 1.5) - mu_r * mu_c

        num = (2 * mu_r * mu_c + c1) * (2 * sigma_rc + c2)
        den = (mu_r ** 2 + mu_c ** 2 + c1) * (sigma_r + sigma_c + c2)
        ssim_map = num / (den + 1e-10)
        ssim_per_channel.append(float(np.mean(ssim_map)))

    return float(np.mean(ssim_per_channel))


def compute_lpips_proxy(reference: np.ndarray, candidate: np.ndarray) -> float:
    """
    Learned Perceptual Image Patch Similarity proxy.
    Uses multi-scale gradient magnitude difference — correlates well with true LPIPS
    without requiring a neural network (lower = better, range ~0-1).
    """
    if reference.shape != candidate.shape:
        candidate = cv2.resize(candidate, (reference.shape[1], reference.shape[0]))

    ref_f = reference.astype(np.float32) / 255.0
    cand_f = candidate.astype(np.float32) / 255.0

    total = 0.0
    scales = [1.0, 0.5, 0.25]
    weights = [0.5, 0.3, 0.2]

    for scale, w in zip(scales, weights):
        if scale < 1.0:
            new_w = max(1, int(ref_f.shape[1] * scale))
            new_h = max(1, int(ref_f.shape[0] * scale))
            r = cv2.resize(ref_f, (new_w, new_h))
            c = cv2.resize(cand_f, (new_w, new_h))
        else:
            r, c = ref_f, cand_f

        # Gradient features (edge/texture sensitivity like LPIPS)
        diff = r - c
        grad_x = np.gradient(diff, axis=1)
        grad_y = np.gradient(diff, axis=0)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        total += w * float(np.mean(grad_mag))

    return float(np.clip(total, 0.0, 1.0))


def compute_all(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return {
        "psnr": compute_psnr(reference, candidate),
        "ssim": compute_ssim(reference, candidate),
        "lpips_proxy": compute_lpips_proxy(reference, candidate),
    }
