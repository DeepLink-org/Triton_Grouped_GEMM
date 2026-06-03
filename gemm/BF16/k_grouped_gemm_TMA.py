import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl

try:
    from ..tma import ensure_triton_tma_allocator
except ImportError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from gemm.tma import ensure_triton_tma_allocator


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 12}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 12}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 6}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 6}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 10}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 10}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 14}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 14}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 16}, num_stages=3, num_warps=8),
    ]


def _get_sm90_benchmark_config(m_dim: int, n_dim: int):
    """Fixed configs for the requested H200/Hopper benchmark shapes."""
    return {
        (1536, 2048): (128, 256, 64, 8, 3, 8),
        (2048, 768): (256, 128, 64, 4, 3, 8),
        (3072, 4096): (128, 256, 64, 1, 3, 8),
        (4096, 1536): (128, 256, 64, 4, 3, 8),
    }.get((m_dim, n_dim))


@triton.jit
def grouped_launch(pid,
                   m, n,
                   block_m: tl.constexpr,
                   block_n: tl.constexpr,
                   group_m: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    remain_pid = pid - group_id * width
    pid_m = group_id * group_m + (remain_pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@triton.autotune(configs=get_cuda_autotune_config(), key=['M_DIM', 'N_DIM'])
@triton.jit
def k_grouped_gemm_kernel(
    A, B, C,
    group_starts,
    group_ends,
    num_groups: tl.constexpr,
    M_DIM: tl.constexpr,
    N_DIM: tl.constexpr,
    K_TOTAL,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr,
    dtype_c: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
    dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
    dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16

    blocks = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_DIM, BLOCK_M)
    num_pid_n = tl.cdiv(N_DIM, BLOCK_N)
    tiles_per_group = num_pid_m * num_pid_n
    num_tiles = tiles_per_group * num_groups

    # A is physical contiguous [K_TOTAL, M_DIM].
    a_desc = tl.make_tensor_descriptor(
        A,
        shape=[K_TOTAL, M_DIM],
        strides=[M_DIM, 1],
        block_shape=[BLOCK_K, BLOCK_M],
    )

    # B only supports physical contiguous [K_TOTAL, N_DIM].
    # The trans_b=True / B[N_DIM, K_TOTAL] path is intentionally removed.
    b_desc = tl.make_tensor_descriptor(
        B,
        shape=[K_TOTAL, N_DIM],
        strides=[N_DIM, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    c_desc = tl.make_tensor_descriptor(
        C,
        shape=[num_groups * M_DIM, N_DIM],
        strides=[N_DIM, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    for tile_id in tl.range(start_pid, num_tiles, blocks):
        group = tile_id // tiles_per_group
        group_start = tl.load(group_starts + group).to(tl.int32)
        group_end = tl.load(group_ends + group).to(tl.int32)
        tokens = group_end - group_start

        local_tile = tile_id - group * tiles_per_group
        if GROUP_M == 1:
            pid_m = local_tile % num_pid_m
            pid_n = local_tile // num_pid_m
        else:
            pid_m, pid_n = grouped_launch(local_tile, M_DIM, N_DIM, BLOCK_M, BLOCK_N, GROUP_M)

        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        offs_k = group_start

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # No group-level padding. We only reduce the real token range
        # [group_start, group_end). Full K blocks use TMA; only the true tail
        # block, when present, uses masked scalar loads.
        full_k_tiles = tokens // BLOCK_K
        tail_tokens = tokens - full_k_tiles * BLOCK_K

        for _ in tl.range(0, full_k_tiles):
            a = a_desc.load([offs_k, offs_am]).to(dtypeA)
            b = b_desc.load([offs_k, offs_bn]).to(dtypeB)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision='tf32x3')
            offs_k += BLOCK_K

        if tail_tokens > 0:
            offs_k_tail = offs_k + tl.arange(0, BLOCK_K)
            offs_m = offs_am + tl.arange(0, BLOCK_M)
            offs_n = offs_bn + tl.arange(0, BLOCK_N)
            k_mask = offs_k_tail < group_end

            a_ptrs = A + offs_k_tail[:, None].to(tl.int64) * M_DIM + offs_m[None, :]
            a = tl.load(a_ptrs, mask=k_mask[:, None], other=0.0)
            b_ptrs = B + offs_k_tail[:, None].to(tl.int64) * N_DIM + offs_n[None, :]
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision='tf32x3')

        c = accumulator.to(dtypeC)
        off_row = group * M_DIM + offs_am
        off_col = offs_bn
        c_desc.store([off_row, off_col], c)


def _dtype_code(dtype: torch.dtype) -> int:
    dtype_mapping = {
        torch.bfloat16: 0,
        torch.float16: 1,
    }
    code = dtype_mapping.get(dtype, -1)
    assert code >= 0, f'data type {dtype} not supported'
    return code


def _check_k_grouped_gemm_inputs(
    A: Tensor,
    B: Tensor,
    size_per_group: Tensor,
    trans_b: bool = False,
) -> Tuple[int, int, int, int]:
    assert not trans_b, 'trans_b=True path has been removed; pass B as contiguous [K_TOTAL, N_DIM] and trans_b=False'
    assert A.is_cuda and B.is_cuda and size_per_group.is_cuda, 'A, B, and size_per_group must be CUDA tensors'
    assert A.dim() == 2, 'A must be 2D'
    assert B.dim() == 2, 'B must be 2D'
    assert A.is_contiguous(), 'A must be contiguous [K_TOTAL, M_DIM]'
    assert B.is_contiguous(), 'B must be contiguous [K_TOTAL, N_DIM]'
    assert A.stride(-1) == 1, 'A must be row-major/K-major physical layout [K_TOTAL, M_DIM]'
    assert B.stride(-1) == 1, 'B must be row-major/K-major physical layout [K_TOTAL, N_DIM]'
    assert size_per_group.dim() == 1, 'size_per_group must be 1D'
    assert size_per_group.dtype in (torch.int32, torch.int64), 'size_per_group must be int32 or int64'

    K_TOTAL, M_DIM = A.shape
    K_B, N_DIM = B.shape
    assert K_TOTAL == K_B, f'A K_TOTAL={K_TOTAL} must equal B K_TOTAL={K_B}'

    _dtype_code(A.dtype)
    _dtype_code(B.dtype)
    assert A.dtype == B.dtype, f'A dtype {A.dtype} and B dtype {B.dtype} should match'

    assert A.data_ptr() % 16 == 0, 'A base pointer must be 16-byte aligned for TMA path'
    assert B.data_ptr() % 16 == 0, 'B base pointer must be 16-byte aligned for TMA path'
    assert (M_DIM * A.element_size()) % 16 == 0, 'A row stride must be 16-byte aligned for TMA path'
    assert (N_DIM * B.element_size()) % 16 == 0, 'B row stride must be 16-byte aligned for TMA path'

    num_groups = int(size_per_group.numel())
    return K_TOTAL, M_DIM, N_DIM, num_groups


def k_grouped_gemm(
    A: Tensor,
    B: Tensor,
    size_per_group: Tensor,
    group_start: Optional[Tensor] = None,
    group_end: Optional[Tensor] = None,
    trans_b: bool = False,
    numSM: int = -1,
    use_fixed_sm90_config: bool = True,
) -> Tensor:
    """Compute per-group A_g.T @ B_g without group-level padding.

    Supported physical layouts:
      A: [K_TOTAL, M_DIM]
      B: [K_TOTAL, N_DIM]
      C: [num_groups, M_DIM, N_DIM]

    The trans_b=True / B[N_DIM, K_TOTAL] path is intentionally removed because
    the measured path was much slower than the B[K_TOTAL, N_DIM] path.
    """
    ensure_triton_tma_allocator()
    K_TOTAL, M_DIM, N_DIM, num_groups = _check_k_grouped_gemm_inputs(A, B, size_per_group, trans_b)

    C = A.new_empty(num_groups, M_DIM, N_DIM)
    if group_start is None or group_end is None:
        group_end = size_per_group.cumsum(0)
        group_start = group_end - size_per_group

    if group_start.dtype != torch.int32:
        group_start = group_start.to(torch.int32)
    if group_end.dtype != torch.int32:
        group_end = group_end.to(torch.int32)

    dtype_a = _dtype_code(A.dtype)
    dtype_b = _dtype_code(B.dtype)
    dtype_c = _dtype_code(C.dtype)

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count if numSM <= 0 else numSM

    def grid(META):
        assert M_DIM % META['BLOCK_M'] == 0, 'Only support M_DIM multiple of BLOCK_M for current TMA load path'
        assert N_DIM % META['BLOCK_N'] == 0, 'Only support N_DIM multiple of BLOCK_N for current TMA load path'
        return (NUM_SMS,)

    fixed_config = None
    if use_fixed_sm90_config and torch.cuda.get_device_capability(A.device)[0] >= 9:
        fixed_config = _get_sm90_benchmark_config(M_DIM, N_DIM)

    if fixed_config is not None:
        block_m, block_n, block_k, group_m, num_stages, num_warps = fixed_config
        k_grouped_gemm_kernel.fn[grid](
            A, B, C,
            group_start,
            group_end,
            num_groups,
            M_DIM,
            N_DIM,
            K_TOTAL,
            dtype_a, dtype_b, dtype_c,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    else:
        k_grouped_gemm_kernel[grid](
            A, B, C,
            group_start,
            group_end,
            num_groups,
            M_DIM,
            N_DIM,
            K_TOTAL,
            dtype_a, dtype_b, dtype_c,
        )
    return C


@dataclass(frozen=True)
class BenchmarkCase:
    model: str
    n: int        # A column dimension / output M_DIM in this script.
    k: int        # B column dimension / output N_DIM in this script.
    total_m: int  # Total reduction/token length K_TOTAL in this script.


BENCHMARK_CASES: List[BenchmarkCase] = [
    # 30B: only trans_b=False path is kept.
    BenchmarkCase('30B', 1536, 2048, 655360),
    BenchmarkCase('30B', 1536, 2048, 655360),
    BenchmarkCase('30B', 2048, 768, 655360),
    BenchmarkCase('30B', 2048, 768, 655360),
    # 235B: only trans_b=False path is kept.
    BenchmarkCase('235B', 3072, 4096, 655360),
    BenchmarkCase('235B', 3072, 4096, 655360),
    BenchmarkCase('235B', 4096, 1536, 655360),
    BenchmarkCase('235B', 4096, 1536, 655360),
]


def _fallback_generate_random_list(num_groups: int, total: int, seed: int = 0) -> List[int]:
    """Generate positive group sizes with exact sum=total."""
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
    """Generate exact-sum positive group sizes, each a multiple of align.

    This keeps the benchmark no-padding at the group level while making TMA
    offsets predictable.
    """
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


def gmm_dw_reference(a: Tensor, b: Tensor, batch_sizes_cpu: Tensor) -> Tensor:
    K_TOTAL, M_DIM = a.shape
    K_B, N_DIM = b.shape
    assert K_TOTAL == K_B

    out = a.new_empty(batch_sizes_cpu.numel(), M_DIM, N_DIM)
    group_end = batch_sizes_cpu.cumsum(0)
    group_start = group_end - batch_sizes_cpu
    for g, (start_t, end_t) in enumerate(zip(group_start, group_end)):
        start = int(start_t.item())
        end = int(end_t.item())
        lhs = a[start:end, :]
        rhs = b[start:end, :]
        out[g] = lhs.T @ rhs
    return out.contiguous()


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
    parser = argparse.ArgumentParser(description='No-padding K-grouped GEMM benchmark, trans_b=False only.')
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
        help='When --random-groups is used, allow exact-sum random sizes without alignment. Tail blocks use masked loads.',
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
        group_end = batch_sizes.cumsum(0).to(torch.int32)
        group_start = (group_end - batch_sizes.to(torch.int32)).to(torch.int32)
        K_TOTAL = int(batch_sizes.sum().item())
        assert K_TOTAL == case.total_m, f'generated K_TOTAL={K_TOTAL}, expected {case.total_m}'
        bad_8, total_groups = _group_boundary_alignment(batch_sizes, 8)
        bad_64, _ = _group_boundary_alignment(batch_sizes, 64)

        a = torch.randn(K_TOTAL, case.n, dtype=dtype, device='cuda').contiguous()
        b = torch.randn(K_TOTAL, case.k, dtype=dtype, device='cuda').contiguous()
        b_layout = 'B[K_TOTAL,N_DIM]'

        out_cublas = None
        out_ref = None
        if not args.skip_check:
            out_ref = gmm_dw_reference(a, b, batch_sizes_cpu)
            out_cublas = torch.empty_like(out_ref) if backend is not None else None
        elif backend is not None:
            out_cublas = torch.empty(args.groups, case.n, case.k, dtype=dtype, device='cuda')

        def triton_call():
            return k_grouped_gemm(
                a,
                b,
                batch_sizes,
                group_start,
                group_end,
                trans_b=False,
                numSM=args.num_sm,
                use_fixed_sm90_config=not args.autotune,
            )

        triton_min_ms, triton_avg_ms, out_triton = benchmark_cuda_event(
            triton_call, warmup=args.warmup, active=args.active
        )

        cublas_min_ms = 0.0
        cublas_avg_ms = 0.0
        if backend is not None:
            def cublas_call():
                backend.gmm(a, b, out_cublas, batch_sizes_cpu, True, False, args.num_sm, False)
                return out_cublas

            cublas_min_ms, cublas_avg_ms, out_cublas = benchmark_cuda_event(
                cublas_call, warmup=args.warmup, active=args.active
            )

        if out_ref is not None:
            out_triton_norm = row_max_normalization(out_triton)
            out_ref_norm = row_max_normalization(out_ref)
            torch.testing.assert_close(out_triton_norm, out_ref_norm, rtol=0.001, atol=0.01)
            if out_cublas is not None:
                out_cublas_norm = row_max_normalization(out_cublas)
                torch.testing.assert_close(out_cublas_norm, out_ref_norm, rtol=0.01, atol=0.01)

        flops = 2 * case.n * case.k * K_TOTAL
        triton_tflops_min = flops / triton_min_ms / 1e9
        triton_tflops_avg = flops / triton_avg_ms / 1e9
        cublas_tflops_min = flops / cublas_min_ms / 1e9 if cublas_min_ms > 0 else 0.0
        speedup = cublas_min_ms / triton_min_ms if cublas_min_ms > 0 else 0.0

        print(
            f'[{case_idx:02d}] model={case.model}, n={case.n}, k={case.k}, '
            f'M={K_TOTAL}, trans_b=False, layout={b_layout}, out_layout=C[G,M,N], '
            f'bad_start_align8={bad_8}/{total_groups}, bad_start_align64={bad_64}/{total_groups}',
            flush=True,
        )
        print(
            f'    Triton GPU time {triton_min_ms:.2f} ms min / {triton_avg_ms:.2f} ms avg, '
            f'{triton_tflops_min:.0f} TFLOP/s min-time / {triton_tflops_avg:.0f} TFLOP/s avg-time',
            flush=True,
        )
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
            'M': K_TOTAL,
            'trans_b': False,
            'layout': b_layout,
            'output_layout': 'C[G,M,N]',
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
            'fixed_config': _get_sm90_benchmark_config(case.n, case.k),
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
