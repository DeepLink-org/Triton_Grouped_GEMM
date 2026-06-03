"""Grouped GEMM layout helpers shared by Triton and optional DeepGEMM paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class GroupedLayout:
    starts: torch.Tensor
    sizes: torch.Tensor
    padded_sizes: torch.Tensor

    @property
    def total_m(self) -> torch.Tensor:
        return self.sizes.sum()

    @property
    def total_padded_m(self) -> torch.Tensor:
        return self.padded_sizes.sum()


def as_group_sizes(
    size_per_group: torch.Tensor | Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    if isinstance(size_per_group, torch.Tensor):
        sizes = size_per_group
        if device is not None:
            sizes = sizes.to(device=device)
        sizes = sizes.to(dtype=dtype)
    else:
        sizes = torch.tensor(size_per_group, device=device, dtype=dtype)
    if sizes.dim() != 1:
        raise ValueError(f"size_per_group must be 1D, got shape {tuple(sizes.shape)}")
    if sizes.numel() == 0:
        raise ValueError("size_per_group must contain at least one group")
    if bool(torch.any(sizes < 0).item()):
        raise ValueError("size_per_group cannot contain negative sizes")
    return sizes.contiguous()


def round_up_to_alignment(sizes: torch.Tensor, alignment: int) -> torch.Tensor:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return torch.div(sizes + alignment - 1, alignment, rounding_mode="floor") * alignment


def make_grouped_layout(size_per_group: torch.Tensor | Sequence[int], block_m: int = 128) -> GroupedLayout:
    sizes = as_group_sizes(size_per_group)
    padded_sizes = round_up_to_alignment(sizes, block_m)
    starts = padded_sizes.cumsum(0) - padded_sizes
    return GroupedLayout(starts=starts.contiguous(), sizes=sizes, padded_sizes=padded_sizes.contiguous())


def make_group_starts(
    size_per_group: torch.Tensor | Sequence[int],
    *,
    padded: bool = False,
    block_m: int = 128,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    sizes = as_group_sizes(size_per_group, dtype=dtype)
    layout_sizes = round_up_to_alignment(sizes, block_m) if padded else sizes
    return (layout_sizes.cumsum(0) - layout_sizes).contiguous()


def make_deep_gemm_grouped_layout(
    size_per_group: torch.Tensor | Sequence[int],
    *,
    block_m: int = 128,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build DeepGEMM's ordinary contiguous grouped layout.

    DeepGEMM expects a vector with one entry per padded M row. Actual rows contain
    their expert id; padded rows contain -1.
    """

    sizes = as_group_sizes(size_per_group, device=device, dtype=torch.int32)
    padded_sizes = round_up_to_alignment(sizes, block_m)
    layout = torch.empty(int(padded_sizes.sum().item()), device=sizes.device, dtype=torch.int32)
    start = 0
    for group, (size, padded_size) in enumerate(zip(sizes.tolist(), padded_sizes.tolist())):
        actual_end = start + size
        padded_end = start + padded_size
        if size:
            layout[start:actual_end] = group
        if padded_size != size:
            layout[actual_end:padded_end] = -1
        start = padded_end
    return layout.contiguous()


def make_deep_gemm_psum_layout(
    size_per_group: torch.Tensor | Sequence[int],
    *,
    block_m: int = 128,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build DeepGEMM's contiguous psum layout.

    Each entry stores the actual end offset of one expert in padded-M
    coordinates. Between entries, starts are aligned up by block_m.
    """

    sizes = as_group_sizes(size_per_group, device=device, dtype=torch.int32)
    padded_sizes = round_up_to_alignment(sizes, block_m)
    ends = torch.empty_like(sizes)
    start = 0
    for group, (size, padded_size) in enumerate(zip(sizes.tolist(), padded_sizes.tolist())):
        ends[group] = start + size
        start += padded_size
    return ends.contiguous()


def validate_total_m(size_per_group: torch.Tensor | Sequence[int], total_m: int) -> None:
    sizes = as_group_sizes(size_per_group, dtype=torch.int64)
    actual = int(sizes.sum().item())
    if actual != total_m:
        raise ValueError(f"sum(size_per_group) must equal M ({total_m}), got {actual}")
