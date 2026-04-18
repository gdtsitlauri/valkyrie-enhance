from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import warnings

from setuptools import setup
import torch
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension, CUDA_HOME


ROOT = Path(__file__).resolve().parent
USE_CUDA = os.environ.get("VALKYRIE_BUILD_CUDA", "1") == "1"
TORCH_CUDA_VERSION = torch.version.cuda


def _detect_nvcc_version(cuda_home: str) -> str | None:
    """Return X.Y version string from nvcc, or None on failure."""
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    for candidate in [str(nvcc), str(nvcc.with_suffix(".exe")), "nvcc"]:
        try:
            out = subprocess.check_output([candidate, "--version"], text=True, stderr=subprocess.STDOUT)
            for token in out.split():
                if token.startswith("V") and token[1:2].isdigit():
                    parts = token[1:].split(".")
                    return ".".join(parts[:2])
        except Exception:
            continue
    return None


if USE_CUDA:
    if TORCH_CUDA_VERSION is None:
        warnings.warn("Torch was built without CUDA support; building CPU-only extension instead.")
        USE_CUDA = False
    elif CUDA_HOME is None:
        warnings.warn("CUDA_HOME is unavailable; building CPU-only extension instead.")
        USE_CUDA = False
    else:
        torch_major_minor = ".".join(TORCH_CUDA_VERSION.split(".")[:2])
        # Try nvcc first for accurate version detection
        detected = _detect_nvcc_version(CUDA_HOME)
        if detected is None:
            # fallback: version.txt
            cuda_version_file = Path(CUDA_HOME) / "version.txt"
            if cuda_version_file.exists():
                text = cuda_version_file.read_text(encoding="utf-8", errors="ignore")
                if "CUDA Version" in text:
                    raw = text.split("CUDA Version", 1)[1].strip().split()[0]
                    detected = ".".join(raw.split(".")[:2])
        if detected and detected != torch_major_minor:
            warnings.warn(
                f"CUDA toolkit {detected} ≠ PyTorch CUDA {TORCH_CUDA_VERSION}; "
                "building CPU-only extension to avoid runtime mismatch."
            )
            USE_CUDA = False

    # On Windows, check MSVC version compatibility with CUDA version.
    # CUDA 11.x supports MSVC up to 19.39 (VS 2022). VS 2026 (MSVC 19.40+)
    # ships an STL that requires CUDA 12.4+.
    if USE_CUDA and sys.platform == "win32":
        cuda_major = int(TORCH_CUDA_VERSION.split(".")[0]) if TORCH_CUDA_VERSION else 0
        try:
            cl_out = subprocess.run(["cl"], capture_output=True, text=True)
            for line in (cl_out.stdout + cl_out.stderr).splitlines():
                if "Version" in line and ("Compiler" in line or "C/C++" in line):
                    for token in line.split():
                        if token[:2].isdigit() and "." in token:
                            parts = token.split(".")
                            msvc_major, msvc_minor = int(parts[0]), int(parts[1])
                            if cuda_major < 12 and (msvc_major, msvc_minor) >= (19, 40):
                                warnings.warn(
                                    f"MSVC {token} (VS 2026+) requires CUDA 12.4+, but PyTorch uses "
                                    f"CUDA {TORCH_CUDA_VERSION}. Building CPU-only extension."
                                )
                                USE_CUDA = False
                            break
                    break
        except Exception:
            pass

sources = [str(ROOT / "valkyrie_native.cpp")]
extension_cls = CppExtension
extra_compile_args: dict[str, list[str]] = {"cxx": ["-O3", "-std=c++17"]}

if USE_CUDA:
    sources.append(str(ROOT / "valkyrie_native_kernel.cu"))
    extension_cls = CUDAExtension
    extra_compile_args["nvcc"] = ["-O3", "--use_fast_math", "-lineinfo", "--allow-unsupported-compiler"]

setup(
    name="valkyrie_native",
    ext_modules=[
        extension_cls(
            name="valkyrie_native",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
