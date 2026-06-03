import torch
from torch import Tensor

import triton
import triton.language as tl

def get_cuda_autotune_config():
    return [
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 4}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 4}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 12}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 12}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 12}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 12}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 6}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 6}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 10}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 10}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 14}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 14}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 16}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 16}, num_stages=3, num_warps=8),
            ]


def _get_sm90_benchmark_config(M: int, N: int):
    return {
        (1536, 2048): (128, 256, 64, 6, 3, 8),
        (2048, 768): (128, 256, 64, 4, 4, 8),
        (3072, 4096): (128, 256, 64, 6, 4, 8),
        (4096, 1536): (128, 256, 32, 4, 4, 8),
    }.get((M, N))


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


@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N'])
@triton.jit
def k_grouped_gemm_kernel(
    A, B, C,
    group_starts,
    group_ends,
    num_groups: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
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

    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n * num_groups

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        
        group = tile_id // (num_pid_m * num_pid_n)
        group_start = tl.load(group_starts + group).to(tl.int32)
        group_end = tl.load(group_ends + group).to(tl.int32)

        id_tmp = tile_id % (num_pid_m * num_pid_n)

        if GROUP_M == 1:
            num_pid_m = tl.cdiv(M, BLOCK_M)
            pid_m = id_tmp % num_pid_m
            pid_n = id_tmp // num_pid_m
        else:
            pid_m, pid_n = grouped_launch(id_tmp, M, N, BLOCK_M, BLOCK_N, GROUP_M)

        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        offs_m = offs_am + tl.arange(0, BLOCK_M)
        offs_n = offs_bn + tl.arange(0, BLOCK_N)
        
        tokens = group_end - group_start

        num_pid_k = tl.cdiv(tokens, BLOCK_K)
        offs_k = group_start + tl.arange(0, BLOCK_K)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kk in tl.range(0, num_pid_k - 1):
            a_ptrs = A + (offs_m[None, :] + offs_k[:, None].to(tl.int64) * M)
            b_ptrs = B + (offs_n[None, :] + offs_k[:, None].to(tl.int64) * N)
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision = "tf32x3")
            offs_k += BLOCK_K
        
        if tokens > 0:
            offs_k_final = group_start + (num_pid_k - 1) * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = A + (offs_m[None, :] + offs_k_final[:, None].to(tl.int64) * M)
            b_ptrs = B + (offs_n[None, :] + offs_k_final[:, None].to(tl.int64) * N)
            tail_mask = offs_k_final[:, None] < group_end
            a = tl.load(a_ptrs, mask=tail_mask, other=0.0)
            b = tl.load(b_ptrs, mask=tail_mask, other=0.0)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision = "tf32x3")    
            c = accumulator.to(dtypeC)
            off_row = offs_m + group * M
            off_col = offs_n
            c_ptrs = C + N * off_row[:, None] + off_col[None, :]
            c_mask = off_col[None, :] < N
            tl.store(c_ptrs, c, mask=c_mask)
            

def k_grouped_gemm(A: Tensor,
                   B: Tensor,
                   size_per_group: torch.Tensor,
                   group_start: Tensor = None,
                   group_end: Tensor = None) -> Tensor:
    assert A.dim() == 2
    assert B.dim() == 2

    K, M = A.shape
    K_, N = B.shape

    assert A.stride(-1) == 1, "Please make sure A is K-major"
    assert B.stride(-1) == 1, "Please make sure B is K-major"
    assert K == K_, "Please make sure that A and B have the same seqlen"
    # assert K * A.element_size() % 128 == 0, "A and B should be 128-byte aligned"
    num_groups = size_per_group.shape[0]

    C = A.new_empty(num_groups, M, N)
    if group_start is None or group_end is None:
        group_end = size_per_group.cumsum(0)
        group_start = group_end - size_per_group

    dtype_mapping = {
        torch.bfloat16: 0,
        torch.float16: 1
    }
    dtype_a = dtype_mapping.get(A.dtype, -1)
    dtype_b = dtype_mapping.get(B.dtype, -1)
    dtype_c = dtype_mapping.get(C.dtype, -1)

    assert dtype_a >= 0, f"data type {A.dtype} not supported" 
    assert dtype_b >= 0, f"data type {B.dtype} not supported" 
    assert dtype_c >= 0, f"data type {C.dtype} not supported" 

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    
    def grid(META):
        assert M % META["BLOCK_M"] == 0, "Only support when M is a multiple of BLOCK_M"
        return (NUM_SMS, )

    fixed_config = None
    if torch.cuda.get_device_capability(A.device)[0] >= 9:
        fixed_config = _get_sm90_benchmark_config(M, N)

    if fixed_config is not None:
        block_m, block_n, block_k, group_m, num_stages, num_warps = fixed_config
        k_grouped_gemm_kernel.fn[grid](
            A, B, C,
            group_start,
            group_end,
            num_groups,
            M,
            N,
            K,
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
            M,
            N,
            K,
            dtype_a, dtype_b, dtype_c,
        )
    # print(f"best config {k_grouped_gemm_kernel.best_config}", flush = True)
    return C


if __name__=='__main__':
    from typing import Tuple
    import random
    
    from utils import generate_random_list, row_max_normalization
    try:
        import grouped_gemm_backend as backend
    except ModuleNotFoundError:
        backend = None
        print(
            "grouped_gemm_backend is not installed; skipping cuBLAS baseline. "
            "Install it with `cd grouped_gemm && python setup.py install` "
            "if you need the cuBLAS comparison.",
            flush=True,
        )
    trans_b = False; trans_a = True

    def gmm_dw(a, b, batch_sizes):
        K, M = a.shape
        K_, N = b.shape

        assert a.stride(-1) == 1, "Please make sure A is K-major"
        assert b.stride(-1) == 1, "Please make sure B is K-major"
        assert K == K_, "Please make sure that A and B have the same seqlen"
        num_groups = batch_sizes.shape[0]

        out = a.new_empty(num_groups, M, N)

        group_end = batch_sizes.cumsum(0) - batch_sizes + batch_sizes
        group_start = batch_sizes.cumsum(0) - batch_sizes
        for g, (start, end) in enumerate(zip(group_start, group_end)):
            rhs = b[start:end, :]
            lhs = a[start:end, :]
            out[g] = lhs.T @ rhs
        return out.contiguous()

    groups = 128; z = groups

    batch_sizes = torch.Tensor(generate_random_list(groups, groups*5120)).cuda().to(torch.int64).abs()
    batch_sizes_cpu = batch_sizes.cpu()
    group_end = batch_sizes.cumsum(0)
    group_start = group_end - batch_sizes
    K = batch_sizes.sum().item()

    def benchmark_cuda_event(fn, warmup=5, active=10):
        out = None
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(active):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        return min(times), sum(times) / len(times), out

    for (m, n) in ((768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)):
        torch.cuda.empty_cache()
        a = torch.randn(K, m, dtype = torch.bfloat16, device = "cuda").view(K, -1)
        b = torch.randn(K, n, dtype = torch.bfloat16, device = "cuda").view(K, -1)
        out_ref = gmm_dw(a, b, batch_sizes.cpu())
        out_cublas = torch.empty_like(out_ref) if backend is not None else None

        def triton_call():
            return k_grouped_gemm(a, b, batch_sizes, group_start, group_end)

        triton_time, triton_avg_time, out_triton = benchmark_cuda_event(triton_call)
        if backend is not None:
            def cublas_call():
                backend.gmm(a, b, out_cublas, batch_sizes_cpu, trans_a, trans_b, -1, False)
                return out_cublas

            cublas_time, cublas_avg_time, out_cublas = benchmark_cuda_event(cublas_call)
        else:
            cublas_time = cublas_avg_time = 0

        # post-process, row normalization
        out_triton = row_max_normalization(out_triton)
        out_ref = row_max_normalization(out_ref)
        if out_cublas is not None:
            out_cublas = row_max_normalization(out_cublas)
    
        torch.testing.assert_close(out_triton, out_ref, rtol = 0.01, atol = 0.01)
        if out_cublas is not None:
            torch.testing.assert_close(out_triton, out_cublas, rtol = 0.01, atol = 0.01)
        torch.cuda.empty_cache()

        print(f"{m = }, {n = }, {K = }")
        flops = 2 * m * n * K
        print(f"    Triton GPU time {triton_time:.2f} ms min / {triton_avg_time:.2f} ms avg, {round(flops / triton_time / 10**9, 0)} tflops")
        if backend is not None:
            print(f"    Cublas GPU time {cublas_time:.2f} ms min / {cublas_avg_time:.2f} ms avg, {round(flops / cublas_time / 10**9, 0)} tflops")
            print(f"    Triton speedup vs Cublas {cublas_time / triton_time: .2f}x")
        else:
            print("    Cublas baseline skipped because grouped_gemm_backend is not installed")
