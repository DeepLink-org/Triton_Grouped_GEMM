"""Backend selection helpers for grouped GEMM kernels."""

from __future__ import annotations

import importlib.util
from enum import Enum
from typing import Optional, Tuple

import torch


class GemmBackend(str, Enum):
    AUTO = "auto"
    TRITON = "triton"
    DEEP_GEMM = "deep_gemm"


def normalize_backend(backend: str | GemmBackend) -> GemmBackend:
    if isinstance(backend, GemmBackend):
        return backend
    backend = {
        "deepgemm": GemmBackend.DEEP_GEMM.value,
        "deep-gemm": GemmBackend.DEEP_GEMM.value,
    }.get(backend, backend)
    try:
        return GemmBackend(backend)
    except ValueError as exc:
        valid = ", ".join(item.value for item in GemmBackend)
        raise ValueError(f"Unsupported GEMM backend {backend!r}; expected one of: {valid}") from exc


def is_deep_gemm_available() -> bool:
    return importlib.util.find_spec("deep_gemm") is not None


def current_cuda_arch() -> Optional[Tuple[int, int]]:
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability()
    except Exception:
        return None


def deep_gemm_supports_current_device() -> bool:
    arch = current_cuda_arch()
    if arch is None:
        return False
    major, _ = arch
    return major in (9, 10)


def select_backend(backend: str | GemmBackend = GemmBackend.AUTO) -> GemmBackend:
    backend = normalize_backend(backend)
    if backend != GemmBackend.AUTO:
        return backend
    if is_deep_gemm_available() and deep_gemm_supports_current_device():
        return GemmBackend.DEEP_GEMM
    return GemmBackend.TRITON


def require_deep_gemm_support() -> None:
    if not is_deep_gemm_available():
        raise RuntimeError("DeepGEMM backend requested, but Python package 'deep_gemm' is not installed")
    if not deep_gemm_supports_current_device():
        arch = current_cuda_arch()
        arch_text = "no CUDA device" if arch is None else f"SM{arch[0]}{arch[1]}"
        raise RuntimeError(f"DeepGEMM backend requested, but current device is unsupported ({arch_text})")
