# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl

try:
    from ..tma import ensure_triton_tma_allocator
except ImportError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from gemm.tma import ensure_triton_tma_allocator

def get_cuda_autotune_config():
    return [
            triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, "GROUP_M": 1}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, "GROUP_M": 18}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 64, 'BLOCK_K': 256, "GROUP_M": 6}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 64, 'BLOCK_K': 256, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 64, 'BLOCK_K': 256, "GROUP_M": 10}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 64, 'BLOCK_K': 256, "GROUP_M": 14}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 64, 'BLOCK_K': 256, "GROUP_M": 18}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 18}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 64, 'BLOCK_K': 256, "GROUP_M": 22}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 22}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64, "GROUP_M": 4}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
            ]


def _get_sm90_benchmark_config(N: int, K: int, trans_b: bool):
    # H200 measured configs for the benchmark shapes in this file.
    common = {
        (1536, 2048): (256, 64, 8, 3, 8),
    }
    return common.get((N, K))


@triton.jit
def grouped_launch(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    remian_pid = pid - group_id * width
    pid_m = group_id * group_m + (remian_pid % group_size)

    pid_n = (pid % width) // group_size

    return pid_m, pid_n

@triton.autotune(configs=get_cuda_autotune_config(), key=['N', 'K'])
@triton.jit
def m_grouped_gemm_bKmajor_kernel(
    A,
    B,
    C,
    pad_starts,
    pad_ends,
    group_starts,
    group_ends,
    m_indices_pad,
    M_pad_ptr,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr, 
    dtype_c: tl.constexpr,
    strideBN: tl.constexpr,
    strideBK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    
    dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
    dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
    dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16

    """gemm fp8 kernel."""
    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    a_desc = tl.make_tensor_descriptor(
        A,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        B,
        shape=[num_groups * N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    c_desc = tl.make_tensor_descriptor(
        C,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        
        # pid_m = tile_id // num_pid_n
        # pid_n = tile_id % num_pid_n

        pid_m, pid_n = grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)

        group = tl.load(m_indices_pad + pid_m)
        pad_off = tl.load(pad_starts + group)

        group_start = (tl.load(group_starts + group) + (pid_m * BLOCK_M - pad_off)).to(tl.int32)
        group_end = tl.load(group_ends + group).to(tl.int32)

        offs_am = group_start
        offs_bn = (group * N + pid_n * BLOCK_N).to(tl.int32)
        offs_k = 0
        
        # b_ptrs_Nmajor = B + ((offs_bn)[None, :] * strideBN + offs_k[:, None] * strideBK)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([offs_am, offs_k]).to(dtypeA)
            b = b_desc.load([offs_bn, offs_k]).to(dtypeB)
            # mma
            accumulator = tl.dot(a, b.T, acc=accumulator, input_precision = "tf32x3")
            offs_k += BLOCK_K
    
        c = accumulator.to(dtypeC)
        offs_cm = group_start
        offs_cn = (pid_n * BLOCK_N).to(tl.int32)
        TMA_condition = (group_start + BLOCK_M <= group_end) & (offs_cn + BLOCK_N <= N)
        if TMA_condition: 
            c_desc.store([offs_cm, offs_cn], c)
        else:
            offs_cm_ = offs_cm + tl.arange(0, BLOCK_M)
            offs_cn_ = offs_cn + tl.arange(0, BLOCK_N)
            c_ptrs = C + N * offs_cm_[:, None].to(tl.int64) + offs_cn_[None, :]
            c_mask = (offs_cm_[:, None] < group_end) & (offs_cn_[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

@triton.autotune(configs=get_cuda_autotune_config(), key=['N', 'K'])
@triton.jit
def m_grouped_gemm_bNmajor_kernel(
    A,
    B,
    C,
    pad_starts,
    pad_ends,
    group_starts,
    group_ends,
    m_indices_pad,
    M_pad_ptr,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr, 
    dtype_c: tl.constexpr,
    strideBN: tl.constexpr,
    strideBK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    
    dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
    dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
    dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16

    """gemm fp8 kernel."""
    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    a_desc = tl.make_tensor_descriptor(
        A,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        B,
        shape=[num_groups * K, N],
        strides=[N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        C,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        
        # pid_m = tile_id // num_pid_n
        # pid_n = tile_id % num_pid_n

        pid_m, pid_n = grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)

        group = tl.load(m_indices_pad + pid_m)
        pad_off = tl.load(pad_starts + group)

        group_start = (tl.load(group_starts + group) + (pid_m * BLOCK_M - pad_off)).to(tl.int32)
        group_end = tl.load(group_ends + group)

        offs_am = group_start
        offs_bn = (pid_n * BLOCK_N).to(tl.int32)
        offs_k = 0
        offs_bk = (group * K).to(tl.int32)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([offs_am, offs_k]).to(dtypeA)
            b = b_desc.load([offs_bk, offs_bn]).to(dtypeB)
            # mma
            accumulator = tl.dot(a, b, acc=accumulator, input_precision = "tf32x3")
            offs_k += BLOCK_K
            offs_bk += BLOCK_K
    
        c = accumulator.to(dtypeC)
        offs_cm = group_start
        offs_cn = (pid_n * BLOCK_N).to(tl.int32)

        TMA_condition = (group_start + BLOCK_M <= group_end) & (offs_cn + BLOCK_N <= N)
        if TMA_condition: 
            c_desc.store([offs_cm, offs_cn], c)
        else:
            offs_cm_ = offs_cm + tl.arange(0, BLOCK_M)
            offs_cn_ = offs_cn + tl.arange(0, BLOCK_N)
            c_ptrs = C + N * offs_cm_[:, None].to(tl.int64) + offs_cn_[None, :]
            c_mask = (offs_cm_[:, None] < group_end) & (offs_cn_[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def repeat_interleave_kernel(
    group_ptr,
    repeats_ptr,
    repeat_cum_ptr,
    output_ptr,
    # BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    repeat = tl.load(repeats_ptr + pid)
    start = tl.load(repeat_cum_ptr + pid) - repeat
    group = tl.load(group_ptr + pid)

    for r in range(repeat):
        tl.store(output_ptr + start + r, group)

def repeat_interleave(
    group_indices: Tensor, 
    repeats: Tensor, 
    repeat_cum: Tensor, 
    m_indices_pad: Tensor, 
) -> None:
    grid = lambda args: (len(repeats), )
    repeat_interleave_kernel[grid](group_indices, repeats, repeat_cum, m_indices_pad)
    return


def _build_m_grouped_metadata(size_per_group: Tensor, block_m: int = 128):
    num_groups = size_per_group.shape[0]
    M = int(size_per_group.sum().item())

    size_per_group_i32 = size_per_group.to(torch.int32)
    m_per_group_padding = torch.div(
        size_per_group_i32 + block_m - 1,
        block_m,
        rounding_mode="floor",
    ) * block_m
    M_pad = m_per_group_padding.sum()

    repeats = (m_per_group_padding // block_m).to(torch.int32)
    m_indices_pad = torch.empty(
        M // block_m + num_groups,
        device=size_per_group.device,
        dtype=torch.int32,
    )
    repeat_interleave(
        torch.arange(num_groups, device=size_per_group.device, dtype=torch.int32),
        repeats,
        repeats.cumsum(0),
        m_indices_pad,
    )

    pad_end = m_per_group_padding.cumsum(0).to(torch.int32)
    pad_start = pad_end - m_per_group_padding
    group_end = size_per_group_i32.cumsum(0)
    group_start = group_end - size_per_group_i32
    return pad_start, pad_end, group_start, group_end, m_indices_pad, M_pad


def _m_grouped_gemm_launch(A: Tensor,
                           B: Tensor,
                           C: Tensor,
                           size_per_group: Tensor,
                           trans_b: bool = False,
                           numSM: int = -1,
                           metadata=None,
                           use_fixed_sm90_config: bool = True) -> Tensor:
    ensure_triton_tma_allocator()
    assert A.dim() == 2
    assert B.dim() == 3
    assert C.dim() == 2

    M, K = A.shape
    if trans_b:
        num_groups, N, BK = B.shape
        strideBN, strideBK = B.stride(1), B.stride(2)
    else:
        num_groups, BK, N = B.shape
        strideBK, strideBN = B.stride(1), B.stride(2)

    assert BK == K, "K of A should be equal to K of B"
    assert C.shape == (M, N), "C should have shape [M, N]"
    assert A.stride(-1) == 1, "Please make sure A is K-major"

    BLOCK_M = 128
    if metadata is None:
        metadata = _build_m_grouped_metadata(size_per_group, BLOCK_M)
    pad_start, pad_end, group_start, group_end, m_indices_pad, M_pad = metadata

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count if numSM <= 0 else numSM

    dtype_mapping = {
        torch.bfloat16: 0,
        torch.float16: 1,
    }
    dtype_a = dtype_mapping.get(A.dtype, -1)
    dtype_b = dtype_mapping.get(B.dtype, -1)
    dtype_c = dtype_mapping.get(C.dtype, -1)
    assert dtype_a >= 0, f"data type {A.dtype} not supported"
    assert dtype_b >= 0, f"data type {B.dtype} not supported"
    assert dtype_c >= 0, f"data type {C.dtype} not supported"

    def grid(META):
        assert (N * B.element_size()) % 16 == 0, "TMA required 16-byte alignment"
        assert (K * B.element_size()) % 16 == 0, "TMA required 16-byte alignment"
        return (NUM_SMS, )

    m_grouped_gemm_kernel = m_grouped_gemm_bKmajor_kernel if trans_b else m_grouped_gemm_bNmajor_kernel
    fixed_config = None
    if use_fixed_sm90_config and torch.cuda.get_device_capability(A.device)[0] >= 9:
        fixed_config = _get_sm90_benchmark_config(N, K, trans_b)

    if fixed_config is not None:
        block_n, block_k, group_m, num_stages, num_warps = fixed_config
        m_grouped_gemm_kernel.fn[grid](
            A,
            B,
            C,
            pad_start,
            pad_end,
            group_start,
            group_end,
            m_indices_pad,
            M_pad,
            num_groups,
            M,
            N,
            K,
            dtype_a,
            dtype_b,
            dtype_c,
            strideBN,
            strideBK,
            BLOCK_M=BLOCK_M,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    else:
        m_grouped_gemm_kernel[grid](
            A,
            B,
            C,
            pad_start,
            pad_end,
            group_start,
            group_end,
            m_indices_pad,
            M_pad,
            num_groups,
            M,
            N,
            K,
            dtype_a,
            dtype_b,
            dtype_c,
            strideBN,
            strideBK,
            BLOCK_M=BLOCK_M,
        )
    return C


@torch.library.custom_op("moe::m_grouped_gemm", mutates_args=())
def m_grouped_gemm(A: Tensor,
                     B: Tensor,
                     size_per_group: torch.Tensor,
                     trans_b: bool = False,
                     numSM: int = -1) -> Tensor:
    M, K = A.shape
    if trans_b:
        _, N, _ = B.shape
    else:
        _, _, N = B.shape
    C = A.new_empty(M, N)
    return _m_grouped_gemm_launch(A, B, C, size_per_group, trans_b, numSM)


@m_grouped_gemm.register_fake
def _(A: Tensor,
        B: Tensor,
        size_per_group: torch.Tensor,
        trans_b: bool = False,
        numSM: int = -1) -> Tensor:
    M, K = A.shape
    if trans_b:
        num_groups, N, BK = B.shape
    else:
        num_groups, BK, N = B.shape
    C = A.new_empty(M, N)
    return C


@dataclass(frozen=True)
class BenchmarkCase:
    model: str
    n: int
    k: int
    total_m: int
    trans_b: bool


BENCHMARK_CASES: List[BenchmarkCase] = [
    # 30B
    BenchmarkCase('30B', 1536, 2048, 655360, True),
    BenchmarkCase('30B', 1536, 2048, 655360, True),
    BenchmarkCase('30B', 1536, 2048, 655360, False),
    BenchmarkCase('30B', 1536, 2048, 655360, False),
    BenchmarkCase('30B', 2048, 768, 655360, True),
    BenchmarkCase('30B', 2048, 768, 655360, True),
    BenchmarkCase('30B', 2048, 768, 655360, False),
    BenchmarkCase('30B', 2048, 768, 655360, False),
    # 235B
    BenchmarkCase('235B', 3072, 4096, 655360, True),
    BenchmarkCase('235B', 3072, 4096, 655360, True),
    BenchmarkCase('235B', 3072, 4096, 655360, False),
    BenchmarkCase('235B', 3072, 4096, 655360, False),
    BenchmarkCase('235B', 4096, 1536, 655360, True),
    BenchmarkCase('235B', 4096, 1536, 655360, True),
    BenchmarkCase('235B', 4096, 1536, 655360, False),
    BenchmarkCase('235B', 4096, 1536, 655360, False),
]


def _fallback_generate_random_list(num_groups: int, total: int, seed: int = 0) -> List[int]:
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    weights = torch.randint(1, 10000, (num_groups,), generator=g, dtype=torch.int64)
    sizes = torch.div(weights * total, weights.sum(), rounding_mode='floor')
    sizes = torch.clamp(sizes, min=1)
    diff = int(total - sizes.sum().item())
    idx = 0
    while diff != 0:
        j = idx % num_groups
        if diff > 0:
            sizes[j] += 1
            diff -= 1
        elif sizes[j] > 1:
            sizes[j] -= 1
            diff += 1
        idx += 1
    return sizes.tolist()


def _fallback_row_max_normalization(x: Tensor) -> Tensor:
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return x / scale


def _load_utils():
    try:
        from utils import generate_random_list, row_max_normalization
        return generate_random_list, row_max_normalization
    except Exception:
        return _fallback_generate_random_list, _fallback_row_max_normalization


def _make_aligned_random_group_sizes(
    num_groups: int,
    total_m: int,
    seed: int,
    align: int = 64,
) -> List[int]:
    assert align > 0
    assert total_m % align == 0, f'total_m={total_m} must be divisible by align={align}'
    total_blocks = total_m // align
    assert total_blocks >= num_groups, 'Need at least one aligned block per group'
    block_sizes = _fallback_generate_random_list(num_groups, total_blocks, seed=seed)
    return [int(x) * align for x in block_sizes]


def make_group_sizes(
    num_groups: int,
    total_m: int,
    seed: int,
    uniform_groups: bool,
    group_align: int = 64,
    random_aligned_groups: bool = True,
) -> Tensor:
    if uniform_groups:
        assert total_m % num_groups == 0, 'uniform_groups requires total_m divisible by num_groups'
        sizes = [total_m // num_groups] * num_groups
    elif random_aligned_groups:
        sizes = _make_aligned_random_group_sizes(num_groups, total_m, seed=seed, align=group_align)
    else:
        generate_random_list, _ = _load_utils()
        try:
            sizes = generate_random_list(num_groups, total_m)
        except TypeError:
            sizes = generate_random_list(num_groups, total_m, seed=seed)
        if sum(int(x) for x in sizes) != total_m:
            sizes = _fallback_generate_random_list(num_groups, total_m, seed=seed)
    return torch.tensor(sizes, device='cuda', dtype=torch.int64)


def _group_boundary_alignment(batch_sizes: Tensor, align: int) -> Tuple[int, int]:
    starts = torch.cumsum(batch_sizes, dim=0) - batch_sizes
    bad = int((starts % align != 0).sum().item())
    return bad, int(starts.numel())


def m_grouped_reference(a: Tensor, b: Tensor, batch_sizes_cpu: Tensor, trans_b: bool) -> Tensor:
    batch_sizes = batch_sizes_cpu.numpy()
    out = []
    start = 0
    for group, size in enumerate(batch_sizes):
        rhs = b[group, :, :].t() if trans_b else b[group, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out).contiguous()


def benchmark_cuda_event(fn, warmup: int = 5, active: int = 10) -> Tuple[float, float, Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(active):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return min(times), sum(times) / len(times), out


def _parse_args():
    parser = argparse.ArgumentParser(description='M-grouped GEMM TMA benchmark.')
    parser.add_argument('--groups', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--active', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--num-sm', type=int, default=-1)
    parser.add_argument('--dtype', choices=['bf16', 'fp16'], default='bf16')
    group_mode = parser.add_mutually_exclusive_group()
    group_mode.add_argument(
        '--uniform-groups',
        dest='uniform_groups',
        action='store_true',
        default=True,
        help='Use equal group sizes. This is the default and is TMA-friendly.',
    )
    group_mode.add_argument(
        '--random-groups',
        dest='uniform_groups',
        action='store_false',
        help='Use random group sizes. By default these are aligned to --group-align.',
    )
    parser.add_argument(
        '--unaligned-random-groups',
        action='store_true',
        help='When --random-groups is used, allow exact-sum random sizes without alignment.',
    )
    parser.add_argument('--group-align', type=int, default=64, help='Alignment, in tokens, for random group sizes.')
    parser.add_argument('--skip-check', action='store_true', help='Skip PyTorch reference correctness check.')
    parser.add_argument('--skip-cublas', action='store_true', help='Skip grouped_gemm_backend baseline even if installed.')
    parser.add_argument('--autotune', action='store_true', help='Disable fixed SM90 configs and let Triton autotune.')
    parser.add_argument('--csv', type=str, default='', help='Optional output CSV path for benchmark summary.')
    parser.add_argument('--only-model', choices=['30B', '235B'], default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16

    _, row_max_normalization = _load_utils()

    try:
        import grouped_gemm_backend as backend
    except ModuleNotFoundError:
        backend = None
        print(
            'grouped_gemm_backend is not installed; skipping cuBLAS baseline. '
            'Install it with `cd grouped_gemm && python setup.py install` if you need comparison.',
            flush=True,
        )

    if args.skip_cublas:
        backend = None

    cases = [case for case in BENCHMARK_CASES if args.only_model is None or case.model == args.only_model]
    results: List[Dict[str, object]] = []

    for case_idx, case in enumerate(cases):
        torch.cuda.empty_cache()
        batch_sizes = make_group_sizes(
            args.groups,
            case.total_m,
            args.seed + case_idx,
            args.uniform_groups,
            group_align=args.group_align,
            random_aligned_groups=not args.unaligned_random_groups,
        )
        batch_sizes_cpu = batch_sizes.cpu()
        M = int(batch_sizes.sum().item())
        assert M == case.total_m, f'generated M={M}, expected {case.total_m}'
        bad_8, total_groups = _group_boundary_alignment(batch_sizes, 8)
        bad_64, _ = _group_boundary_alignment(batch_sizes, 64)
        metadata = _build_m_grouped_metadata(batch_sizes)

        a = torch.randn(M, case.k, dtype=dtype, device='cuda').contiguous()
        if case.trans_b:
            b = torch.randn(args.groups, case.n, case.k, dtype=dtype, device='cuda').contiguous()
            b_layout = 'B[G,N,K]'
        else:
            b = torch.randn(args.groups, case.k, case.n, dtype=dtype, device='cuda').contiguous()
            b_layout = 'B[G,K,N]'

        out_ref = None
        if not args.skip_check:
            out_ref = m_grouped_reference(a, b, batch_sizes_cpu, case.trans_b)
            out_triton = torch.empty_like(out_ref)
            out_cublas = torch.empty_like(out_ref) if backend is not None else None
        else:
            out_triton = torch.empty(M, case.n, dtype=dtype, device='cuda')
            out_cublas = torch.empty_like(out_triton) if backend is not None else None

        def triton_call():
            return _m_grouped_gemm_launch(
                a,
                b,
                out_triton,
                batch_sizes,
                case.trans_b,
                args.num_sm,
                metadata=metadata,
                use_fixed_sm90_config=not args.autotune,
            )

        triton_min_ms, triton_avg_ms, out_triton = benchmark_cuda_event(
            triton_call, warmup=args.warmup, active=args.active
        )

        cublas_min_ms = 0.0
        cublas_avg_ms = 0.0
        if backend is not None:
            def cublas_call():
                backend.gmm(a, b, out_cublas, batch_sizes_cpu, False, case.trans_b, args.num_sm, False)
                return out_cublas

            cublas_min_ms, cublas_avg_ms, out_cublas = benchmark_cuda_event(
                cublas_call, warmup=args.warmup, active=args.active
            )

        if out_ref is not None:
            out_triton_norm = row_max_normalization(out_triton)
            out_ref_norm = row_max_normalization(out_ref)
            torch.testing.assert_close(out_triton_norm, out_ref_norm, rtol=1e-02, atol=1e-02)
            if out_cublas is not None:
                out_cublas_norm = row_max_normalization(out_cublas)
                torch.testing.assert_close(out_cublas_norm, out_ref_norm, rtol=5e-03, atol=5e-03)

        flops = 2 * M * case.n * case.k
        triton_tflops_min = flops / triton_min_ms / 1e9
        triton_tflops_avg = flops / triton_avg_ms / 1e9
        cublas_tflops_min = flops / cublas_min_ms / 1e9 if cublas_min_ms > 0 else 0.0
        speedup = cublas_min_ms / triton_min_ms if cublas_min_ms > 0 else 0.0
        best_config = None
        if args.autotune:
            kernel = m_grouped_gemm_bKmajor_kernel if case.trans_b else m_grouped_gemm_bNmajor_kernel
            best_config = getattr(kernel, 'best_config', None)

        print(
            f'[{case_idx:02d}] model={case.model}, n={case.n}, k={case.k}, '
            f'M={M}, trans_b={case.trans_b}, layout={b_layout}, out_layout=C[M,N], '
            f'bad_start_align8={bad_8}/{total_groups}, bad_start_align64={bad_64}/{total_groups}',
            flush=True,
        )
        print(
            f'    Triton GPU time {triton_min_ms:.2f} ms min / {triton_avg_ms:.2f} ms avg, '
            f'{triton_tflops_min:.0f} TFLOP/s min-time / {triton_tflops_avg:.0f} TFLOP/s avg-time',
            flush=True,
        )
        if best_config is not None:
            print(f'    Triton autotune best_config={best_config}', flush=True)
        if backend is not None:
            print(
                f'    Cublas GPU time {cublas_min_ms:.2f} ms min / {cublas_avg_ms:.2f} ms avg, '
                f'{cublas_tflops_min:.0f} TFLOP/s; Triton speedup {speedup:.2f}x '
                f'(baseline_layout={b_layout})',
                flush=True,
            )
        else:
            print('    Cublas baseline skipped because grouped_gemm_backend is not installed', flush=True)

        results.append({
            'case_idx': case_idx,
            'model': case.model,
            'n': case.n,
            'k': case.k,
            'M': M,
            'trans_b': case.trans_b,
            'layout': b_layout,
            'output_layout': 'C[M,N]',
            'groups': args.groups,
            'dtype': args.dtype,
            'triton_min_ms': triton_min_ms,
            'triton_avg_ms': triton_avg_ms,
            'triton_tflops_min_time': triton_tflops_min,
            'triton_tflops_avg_time': triton_tflops_avg,
            'cublas_min_ms': cublas_min_ms,
            'cublas_avg_ms': cublas_avg_ms,
            'cublas_tflops_min_time': cublas_tflops_min,
            'speedup_vs_cublas': speedup,
            'fixed_config': _get_sm90_benchmark_config(case.n, case.k, case.trans_b),
            'autotune_best_config': str(best_config) if best_config is not None else '',
            'cublas_layout': b_layout if backend is not None else '',
            'bad_start_align8': bad_8,
            'bad_start_align64': bad_64,
        })

        del a, b, out_triton, out_ref, out_cublas
        torch.cuda.empty_cache()

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f'Wrote CSV summary to {csv_path}', flush=True)


if __name__ == '__main__':
    main()
