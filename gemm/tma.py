"""Triton TMA runtime helpers."""

from __future__ import annotations

import threading

import torch
import triton

_ALLOCATOR_LOCK = threading.Lock()
_ALLOCATOR_SET = False


def _triton_tma_allocator(size: int, alignment: int, stream: int | None):
    del alignment, stream
    return torch.empty(size, device=torch.cuda.current_device(), dtype=torch.int8)


def ensure_triton_tma_allocator() -> None:
    """Install the runtime allocator required by Triton tensor descriptors."""

    global _ALLOCATOR_SET
    if _ALLOCATOR_SET:
        return
    with _ALLOCATOR_LOCK:
        if not _ALLOCATOR_SET:
            triton.set_allocator(_triton_tma_allocator)
            _ALLOCATOR_SET = True

