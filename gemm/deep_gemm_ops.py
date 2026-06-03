"""Optional DeepGEMM entrypoints.

These wrappers keep DeepGEMM as an optional dependency. Importing this module
does not import deep_gemm; the package is loaded only when a wrapper is called.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .layout import make_deep_gemm_grouped_layout, round_up_to_alignment


def _require_deep_gemm():
    try:
        import deep_gemm  # type: ignore
    except ImportError as exc:
        raise RuntimeError("DeepGEMM backend requested, but Python package 'deep_gemm' is not installed") from exc
    return deep_gemm


def m_grouped_bf16_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    size_per_group: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    grouped_layout: Optional[torch.Tensor] = None,
    block_m: int = 128,
    use_psum_layout: bool = False,
) -> torch.Tensor:
    """Run DeepGEMM BF16 grouped GEMM for NT layout.

    A has shape [M, K], B has shape [num_groups, N, K], and output is [M, N].
    A is expected to include padded expert rows. size_per_group stores actual
    rows per expert; grouped_layout marks actual rows with expert ids and padded
    rows with -1 unless provided explicitly.
    """

    deep_gemm = _require_deep_gemm()
    if a.dim() != 2 or b.dim() != 3:
        raise ValueError("expected a shape [M, K] and b shape [num_groups, N, K]")
    m, k = a.shape
    _, n, bk = b.shape
    if bk != k:
        raise ValueError(f"K mismatch: a has K={k}, b has K={bk}")
    expected_m = int(round_up_to_alignment(size_per_group.to(dtype=torch.int32), block_m).sum().item())
    if m != expected_m:
        raise ValueError(f"A must include padded expert rows: expected M={expected_m}, got M={m}")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if grouped_layout is None:
        grouped_layout = make_deep_gemm_grouped_layout(size_per_group, block_m=block_m, device=a.device)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, out, grouped_layout, use_psum_layout=use_psum_layout)
    return out


def m_grouped_fp8_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    size_per_group: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    grouped_layout: Optional[torch.Tensor] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    block_m: int = 128,
    use_psum_layout: bool = False,
) -> torch.Tensor:
    """Run DeepGEMM FP8 grouped GEMM for NT layout.

    A is a pair (a_fp8, a_scale), B is a pair (b_fp8, b_scale). Scale tensors
    must already be in the layout expected by the installed DeepGEMM version.
    """

    deep_gemm = _require_deep_gemm()
    a_tensor, _ = a
    b_tensor, _ = b
    if a_tensor.dim() != 2 or b_tensor.dim() != 3:
        raise ValueError("expected a[0] shape [M, K] and b[0] shape [num_groups, N, K]")
    m, k = a_tensor.shape
    _, n, bk = b_tensor.shape
    if bk != k:
        raise ValueError(f"K mismatch: a has K={k}, b has K={bk}")
    expected_m = int(round_up_to_alignment(size_per_group.to(dtype=torch.int32), block_m).sum().item())
    if m != expected_m:
        raise ValueError(f"A must include padded expert rows: expected M={expected_m}, got M={m}")
    if out is None:
        out = torch.empty((m, n), device=a_tensor.device, dtype=torch.bfloat16)
    if grouped_layout is None:
        grouped_layout = make_deep_gemm_grouped_layout(size_per_group, block_m=block_m, device=a_tensor.device)
    kwargs = {}
    if recipe_a is not None:
        kwargs["recipe_a"] = recipe_a
    if recipe_b is not None:
        kwargs["recipe_b"] = recipe_b
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, out, grouped_layout, use_psum_layout=use_psum_layout, **kwargs)
    return out
