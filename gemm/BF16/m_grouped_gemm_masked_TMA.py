# Copyright (c) OpenMMLab. lhsll rights reserved.
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
            ]

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
def m_grouped_gemm_masked_kernel(
    lhs,
    rhs,
    C,
    masked_m,
    pad_mask_starts,
    m_indices_pad,
    M_pad_masked_ptr,
    num_groups: tl.constexpr,
    expected_m: tl.constexpr,
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

    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    M_pad_masked = tl.load(M_pad_masked_ptr)
    num_pid_m = tl.cdiv(M_pad_masked, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    a_desc = tl.make_tensor_descriptor(
        lhs,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        rhs,
        shape=[num_groups * N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        
        pid_m, pid_n = grouped_launch(tile_id, M_pad_masked, N, BLOCK_M, BLOCK_N, GROUP_M)

        group = tl.load(m_indices_pad + pid_m)
        masked_pad_off = tl.load(pad_mask_starts + group)

        group_start = (group * expected_m + (pid_m * BLOCK_M - masked_pad_off)).to(tl.int32)
        group_end = group_start + tl.load(masked_m + group)
        offs_am = group_start
        offs_bn = (group * N + pid_n * BLOCK_N).to(tl.int32)
        offs_k = 0

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            # load ab
            a = a_desc.load([offs_am, offs_k]).to(dtypeA)
            b = b_desc.load([offs_bn, offs_k]).to(dtypeB)
            # mma
            accumulator = tl.dot(a, b.T, acc=accumulator, input_precision = "tf32x3")
            offs_k += BLOCK_K

        c = accumulator.to(dtypeC)
        offs_cm = group_start
        offs_cn = (pid_n * BLOCK_N).to(tl.int32)


        offs_cm_ = offs_cm + tl.arange(0, BLOCK_M)
        offs_cn_ = offs_cn + tl.arange(0, BLOCK_N)
        c_ptrs = C + N * offs_cm_[:, None] + offs_cn_[None, :]
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

# @torch.compile
def m_grouped_gemm_masked(lhs: Tensor,
                     rhs: Tensor,
                     out: torch.Tensor,
                     masked_m: torch.Tensor,
                     expected_m: int,
                     trans_b: bool = True) -> Tensor:
    """
    Perform a grouped GEMM (masked format) with BF16 inputs and BF16 output.

    Requirements:
        LHS, RHS, and output tensors must be in contiguous format.

    Arguments:
        lhs: an BF16 tensor shape `[num_groups, m_max, k]`,
        rhs: an BF16 tensor of shape `[num_groups, n, k]`.
        out: the BF16 output tensor of shape `[num_groups, m_max, n]`, representing the result.
        masked_m: a tensor of shape `[num_groups]`, `masked_m[i]` records actual rows of the `lhs[i]` matrix to compute
            in the i-th group.
        expected_m: a value hint (which is a value on CPU) for the M expectation of each batch,
            correctly setting this value may lead to better performance.
    """
    stream = torch.cuda.Stream(lhs.device)
    with torch.cuda.stream(stream):
        ensure_triton_tma_allocator()
        assert lhs.dim() == 2
        assert rhs.dim() == 3

        M, K = lhs.shape
        
        assert lhs.stride(-1) == 1, "Please make sure lhs is K-major"
        if trans_b:
            num_groups, N, rhsK = rhs.shape
            strideBN, strideBK = rhs.stride(1), rhs.stride(2)
        else:
            num_groups, rhsK, N = rhs.shape
            strideBK, strideBN = rhs.stride(1), rhs.stride(2)

        assert rhsK == K, "K of lhs should be equal to K of rhs"
        # C = lhs.new_empty(M, N)

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        dtype_mapping = {
            torch.bfloat16: 0,
            torch.float16: 1
        }
        dtype_a = dtype_mapping.get(lhs.dtype, -1)
        dtype_b = dtype_mapping.get(rhs.dtype, -1)
        dtype_c = dtype_mapping.get(out.dtype, -1)

        def grid(META):
            assert (N * rhs.element_size()) % 16 == 0, "TMA required 16-byte alignment"
            assert (K * rhs.element_size()) % 16 == 0, "TMA required 16-byte alignment"
            assert M % BLOCK_M == 0, "Only support when M is a multiple of BLOCK_M"
            return (NUM_SMS, )

        BLOCK_M = 128

        masked_per_group_padding = triton.cdiv(masked_m, BLOCK_M) * BLOCK_M
        M_pad_masked = masked_per_group_padding.sum()

        repeats = (masked_per_group_padding // BLOCK_M).to(torch.int32)
        m_indices_pad = torch.empty(M // BLOCK_M + num_groups, device = masked_m.device, dtype = torch.int64)
        repeat_interleave(torch.arange(num_groups, device=lhs.device).to(torch.int32), repeats, repeats.cumsum(0), m_indices_pad)
        
        pad_mask_start = masked_per_group_padding.cumsum(0) - masked_per_group_padding

        m_grouped_gemm_masked_kernel[grid](
                lhs,
                rhs,
                out,
                masked_m,
                pad_mask_start,
                m_indices_pad,
                M_pad_masked,
                num_groups,
                expected_m,
                M,
                N,
                K,
                dtype_a, dtype_b, dtype_c,
                strideBN,
                strideBK,
                BLOCK_M=BLOCK_M,
            )
        # print(f"best config {m_grouped_gemm_masked_kernel.best_config}", flush = True)
        return


if __name__=='__main__':
    from typing import Tuple
    import random
    
    from torch.profiler import ProfilerActivity, profile, record_function
    from torch.library import triton_op, wrap_triton

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

    def gmm(a, b, batch_sizes, trans_b=False):
        batch_sizes = batch_sizes.numpy()

        out = []
        start = 0
        for i, size in enumerate(batch_sizes):
            rhs = b[i, :, :].t() if trans_b else b[i, :, :]
            out.append(a[start:start + size, :] @ rhs)
            start += size
        return torch.cat(out)


    groups = 128; z = groups
    trans_b = True; print(f"{trans_b = }")
    if trans_b == False:
        raise NotImplementedError("Not support when trans_b != True")
    expected_m = 4096
    device = "cuda"
    batch_sizes = expected_m * torch.ones(groups, device = device, dtype = torch.int32)

    masked_m = torch.Tensor(generate_random_list(groups, groups*1024)).to(device).to(torch.int64).abs()
    masked_m[0:127:5] = 0
    # masked_m = batch_sizes
    masked_m = torch.where(masked_m > batch_sizes, batch_sizes, masked_m)
    # print(f"{masked_m = }")
    
    batch_sizes_cpu = batch_sizes.cpu()
    M = batch_sizes.sum().item()
    M_masked = masked_m.sum().item()
    masked_m_cpu = masked_m.cpu()
    active_groups = [g for g, size in enumerate(masked_m_cpu.tolist()) if size > 0]

    for (n, k) in ((768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)):
        torch.cuda.empty_cache()
        a = torch.randn(M, k, dtype = torch.bfloat16, device = device).view(-1, k).requires_grad_(True)
        b = torch.randn(z, n, k, dtype = torch.bfloat16, device = device) if trans_b else torch.randn(z, k, n, dtype = torch.bfloat16, device = device).requires_grad_(True)
        out_ref = gmm(a, b, batch_sizes.cpu(), trans_b)
        out_triton = torch.empty((M, n), dtype = torch.bfloat16, device = device)
        if backend is not None:
            packed_a = torch.cat([
                a[g * expected_m:g * expected_m + int(masked_m_cpu[g].item()), :]
                for g in active_groups
            ])
            active_group_tensor = torch.tensor(active_groups, device=device, dtype=torch.long)
            packed_b = b.index_select(0, active_group_tensor)
            packed_sizes_cpu = masked_m_cpu[active_groups]
            out_cublas = torch.empty((M_masked, n), dtype = torch.bfloat16, device = device)
        else:
            out_cublas = None

        from pathlib import Path
        script_path = Path(__file__).resolve()
        parent_dir = script_path.parent.parent
        from torch.profiler import ProfilerActivity, profile, record_function
        trace_file = f"{parent_dir}/trace/gmm_triton_masked_N{n}_K{k}"  + ".json"
        import os
        Path(os.path.join(parent_dir, "trace")) .mkdir(parents=True, exist_ok=True)
        active_ = 3
        def trace_handler(prof):
            prof.export_chrome_trace(trace_file)
        with profile(
            activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=3,
                active=active_,
                repeat=0),
            on_trace_ready=trace_handler,
            with_modules = True,
            record_shapes=True,) as prof:
            for i in range(4+active_):
                with record_function(f"Triton_record"):
                    m_grouped_gemm_masked(a, b, out_triton, masked_m, expected_m, trans_b)
                torch.cuda.synchronize(device = device)
                if backend is not None:
                    with record_function(f"Cublas_record"):
                        backend.gmm(packed_a, packed_b, out_cublas, packed_sizes_cpu, False, trans_b, -1, False)
                    torch.cuda.synchronize(device = device)
                prof.step()

        # post-process, row normalization
        out_triton = row_max_normalization(out_triton)
        out_ref = row_max_normalization(out_ref)
        if out_cublas is not None:
            out_cublas = row_max_normalization(out_cublas)

        group_end = batch_sizes.cumsum(0) - batch_sizes + masked_m
        group_start = batch_sizes.cumsum(0) - batch_sizes
        for g in range(groups):
            torch.testing.assert_close(out_triton[group_start[g]:group_end[g], :], out_ref[group_start[g]:group_end[g], :], rtol = 0.001, atol = 0.005)
        if out_cublas is not None:
            active_ref = torch.cat([
                out_ref[g * expected_m:g * expected_m + int(masked_m_cpu[g].item()), :]
                for g in active_groups
            ])
            torch.testing.assert_close(out_cublas, active_ref, rtol = 0.01, atol = 0.01)
        print(f"{n = }, {k = }, {M_masked = }")
        

        import json
        with open(trace_file, "r") as file:
            data = json.load(file)

        def process_events(data, record_function):
            func_dict = {}
            for event in data["traceEvents"]:
                if event["name"] == record_function and "gpu_user_annotation" in event["cat"]:
                    start = event["ts"]
                    end = start + event["dur"]
                    cpu_id = event['args']['External id']
                    if cpu_id not in func_dict:
                        func_dict[cpu_id] = {"start": start, "end": end}
                    else:
                        func_dict[cpu_id]["start"] = min(start, func_dict[cpu_id]["start"])
                        func_dict[cpu_id]["end"] = max(end, func_dict[cpu_id]["end"])
            durations = []
            for cpu_id in func_dict:
                duration = (func_dict[cpu_id]["end"] - func_dict[cpu_id]["start"]) / 1000
                func_dict[cpu_id]["dur"] = duration
                durations.append(duration)
            func_time = sum(durations) / len(durations) if durations else 0
            return func_dict, func_time

        triton_dict, triton_time = process_events(data, "Triton_record")
        flops = 2 * M_masked * n * k
        print(f"    Triton call Elapsed time {round((triton_time), 2)} ms, {round(flops / triton_time / 10**9, 0)} tflops")
        if backend is not None:
            cublas_dict, cublas_time = process_events(data, "Cublas_record")
            print(f"    Cublas packed-active call Elapsed time {round((cublas_time), 2)} ms, {round(flops / cublas_time / 10**9, 0)} tflops")
            print(f"    Triton speedup vs Cublas {cublas_time / triton_time: .2f}x")
        else:
            print("    Cublas baseline skipped because grouped_gemm_backend is not installed")
