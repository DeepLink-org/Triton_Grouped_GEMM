# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

import triton
import triton.language as tl

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
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        
        # pid_m = tile_id // num_pid_n
        # pid_n = tile_id % num_pid_n

        pid_m, pid_n = grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)

        group = tl.load(m_indices_pad + pid_m)
        pad_off = tl.load(pad_starts + group)

        group_start = (tl.load(group_starts + group) + (pid_m * BLOCK_M - pad_off)).to(tl.int64)
        group_end = tl.load(group_ends + group).to(tl.int64)

        offs_am = group_start + tl.arange(0, BLOCK_M)
        offs_bn = (group * N + pid_n * BLOCK_N).to(tl.int64) + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            a_ptrs = A + ((K * offs_am)[:, None] + offs_k[None, :].to(tl.int64))
            b_ptrs = B + ((K * offs_bn)[:, None] + offs_k[None, :].to(tl.int64))
            maskA = (offs_am[:, None] < group_end) & (offs_k[None, :] < K)
            maskB = (offs_bn[:, None] < (group + 1) * N) & (offs_k[None, :] < K)
            
            a = tl.load(a_ptrs, mask=maskA, other=0.0)
            b = tl.load(b_ptrs, mask=maskB, other=0.0)
            # mma
            accumulator = tl.dot(a, b.T, acc=accumulator, input_precision = "tf32x3")
            offs_k += BLOCK_K
    
        c = accumulator.to(dtypeC)
        offs_cm = group_start + tl.arange(0, BLOCK_M)
        offs_cn = (pid_n * BLOCK_N).to(tl.int64) + tl.arange(0, BLOCK_N)
        c_ptrs = C + N * offs_cm[:, None].to(tl.int64) + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < group_end) & (offs_cn[None, :] < N)
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
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        
        pid_m, pid_n = grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)

        group = tl.load(m_indices_pad + pid_m)
        pad_off = tl.load(pad_starts + group)

        group_start = (tl.load(group_starts + group) + (pid_m * BLOCK_M - pad_off)).to(tl.int64)
        group_end = tl.load(group_ends + group)

        offs_am = group_start.to(tl.int64) + tl.arange(0, BLOCK_M)
        offs_bn = (pid_n * BLOCK_N).to(tl.int64) + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
        offs_bk = (group * K).to(tl.int64) + tl.arange(0, BLOCK_K).to(tl.int64)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            a_ptrs = A + ((K * offs_am)[:, None] + offs_k[None, :].to(tl.int64))
            b_ptrs = B + ((N * offs_bk)[:, None].to(tl.int64) + (offs_bn)[None, :])
            maskA = (offs_am[:, None] < group_end) & (offs_k[None, :] < K)
            maskB = (offs_bn[None, :] < N) & (offs_bk[:, None] < (group + 1) * K)
            
            a = tl.load(a_ptrs, mask=maskA, other=0.0)
            b = tl.load(b_ptrs, mask=maskB, other=0.0)
            # mma
            accumulator = tl.dot(a, b, acc=accumulator, input_precision = "tf32x3")
            offs_k += BLOCK_K
            offs_bk += BLOCK_K
    
        c = accumulator.to(dtypeC)
        offs_cm = group_start.to(tl.int64) + tl.arange(0, BLOCK_M)
        offs_cn = (pid_n * BLOCK_N).to(tl.int64) + tl.arange(0, BLOCK_N)

        c_ptrs = C + N * offs_cm[:, None].to(tl.int64) + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < group_end) & (offs_cn[None, :] < N)
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


@torch.library.custom_op("moe::m_grouped_gemm", mutates_args=())
def m_grouped_gemm(A: Tensor,
                     B: Tensor,
                     size_per_group: torch.Tensor,
                     trans_b: bool = False,
                     numSM: int = -1) -> Tensor:
    stream = torch.cuda.Stream(A.device)
    with torch.cuda.stream(stream):
        assert A.dim() == 2
        assert B.dim() == 3

        M, K = A.shape
        
        assert A.stride(-1) == 1, "Please make sure A is K-major"
        if trans_b:
            num_groups, N, BK = B.shape
            strideBN, strideBK = B.stride(1), B.stride(2)
        else:
            num_groups, BK, N = B.shape
            strideBK, strideBN = B.stride(1), B.stride(2)

        assert BK == K, "K of A should be equal to K of B"
        C = A.new_empty(M, N)

        BLOCK_M = 128
        m_per_group_padding = triton.cdiv(size_per_group, BLOCK_M) * BLOCK_M
        M_pad = m_per_group_padding.sum()

        repeats = (m_per_group_padding // BLOCK_M).to(torch.int32)
        m_indices_pad = torch.empty(M // BLOCK_M + num_groups, device = size_per_group.device, dtype = torch.int64)
        repeat_interleave(torch.arange(num_groups, device='cuda').to(torch.int32), repeats, repeats.cumsum(0), m_indices_pad)

        pad_start = m_per_group_padding.cumsum(0) - m_per_group_padding
        pad_end = m_per_group_padding.cumsum(0)

        group_end = size_per_group.cumsum(0) - size_per_group + size_per_group
        group_start = size_per_group.cumsum(0) - size_per_group


        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count if numSM <= 0 else numSM

        dtype_mapping = {
            torch.bfloat16: 0,
            torch.float16: 1
        }
        dtype_a = dtype_mapping.get(A.dtype, -1)
        dtype_b = dtype_mapping.get(B.dtype, -1)
        dtype_c = dtype_mapping.get(C.dtype, -1)

        def grid(META):
            return (NUM_SMS, )

        m_grouped_gemm_kernel = m_grouped_gemm_bKmajor_kernel if trans_b else m_grouped_gemm_bNmajor_kernel
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
                M,
                N,
                K,
                dtype_a, dtype_b, dtype_c,
                strideBN,
                strideBK,
                BLOCK_M=BLOCK_M,
            )
        # print(f"best config {m_grouped_gemm_kernel.best_config}", flush = True)
        return C


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


if __name__=='__main__':
    from typing import Tuple
    import random
    
    from torch.profiler import ProfilerActivity, profile, record_function
    from torch.library import triton_op, wrap_triton
    import grouped_gemm_backend as backend
    use_cutlass = 0; trans_a = False

    from utils import generate_random_list, row_max_normalization

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
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*4096)).cuda().to(torch.int64)
    batch_sizes_cpu = batch_sizes.cpu()
    M = batch_sizes.sum().item()

    for (n, k) in ((768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)):
        # for (n, k) in ((4096, 1536),):
        torch.cuda.empty_cache()
        a = torch.randn(M, k, dtype = torch.bfloat16, device = "cuda").view(-1, k).requires_grad_(True)
        b = torch.randn(z, n, k, dtype = torch.bfloat16, device = "cuda") if trans_b else torch.randn(z, k, n, dtype = torch.bfloat16, device = "cuda").requires_grad_(True)
        out_ref = gmm(a, b, batch_sizes.cpu(), trans_b)
        out_cublas = out_ref.new_empty(out_ref.size())
        out_cutlass = out_ref.new_empty(out_ref.size())

        for i in range(3):
            out_triton = m_grouped_gemm(a, b, batch_sizes, trans_b)
            backend.gmm(a, b, out_cublas, batch_sizes_cpu, False, trans_b, -1, False)
            # backend.gmm(a, b, out_cutlass, batch_sizes, False, trans_b, -1, True)

        from pathlib import Path
        script_path = Path(__file__).resolve()
        parent_dir = script_path.parent.parent
        from torch.profiler import ProfilerActivity, profile, record_function
        trace_file = f"{parent_dir}/trace/gmm_triton_cublas_cutlass_N{n}_K{k}"  + ".json"
        import os
        Path(os.path.join(parent_dir, "trace")) .mkdir(parents=True, exist_ok=True)

        def trace_handler(prof):
            prof.export_chrome_trace(trace_file)
        activate_ = 30
        def trace_handler(prof):
            prof.export_chrome_trace(trace_file)
        with profile(
            activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=activate_,
                repeat=0),
            on_trace_ready=trace_handler,
            with_modules = True,
            record_shapes=True,) as prof:
            for i in range(10+activate_):
                with record_function(f"Triton_record"):
                    out_triton = m_grouped_gemm(a, b, batch_sizes, trans_b)
                with record_function(f"Cublas_record"):
                    backend.gmm(a, b, out_cublas, batch_sizes_cpu, False, trans_b, -1, False)
                # with record_function(f"Cutlass_record"):
                #     backend.gmm(a, b, out_cutlass, batch_sizes, False, trans_b, -1, True)
                prof.step()

        # post-process, row normalization
        out_triton = row_max_normalization(out_triton)
        out_cublas = row_max_normalization(out_cublas)
        # out_cutlass = row_max_normalization(out_cutlass)
        out_ref = row_max_normalization(out_ref)

        torch.testing.assert_close(out_triton, out_ref, rtol = 1e-02, atol = 1e-02)
        torch.testing.assert_close(out_cublas, out_ref, rtol = 5e-03, atol = 5e-03)
        # torch.testing.assert_close(out_cutlass, out_ref, rtol = 1e-02, atol = 1e-02)

        print(f"{n = }, {k = }, {M = }")
        
        import json
        with open(trace_file, "r") as file:
            data = json.load(file)

        def process_events(data, record_function):
            func_dict = {}
            
            # Process each event to collect cublas records
            for event in data["traceEvents"]:
                if event["name"] == record_function and "gpu_user_annotation" in event["cat"]:
                    start = event["ts"]
                    end = start + event["dur"]
                    # import pdb; pdb.set_trace()
                    cpu_id = event['args']['External id']
                    
                    if cpu_id not in func_dict:
                        # Initialize if id doesn't exist
                        func_dict[cpu_id] = {"start": start, "end": end}
                    else:
                        # Update start and end if id exists
                        func_dict[cpu_id]["start"] = min(start, func_dict[cpu_id]["start"])
                        func_dict[cpu_id]["end"] = max(end, func_dict[cpu_id]["end"])
            
            # Calculate duration for each cublas event in microseconds
            durations = []
            for cpu_id in func_dict:
                duration = (func_dict[cpu_id]["end"] - func_dict[cpu_id]["start"]) / 1000  # Convert to milliseconds
                func_dict[cpu_id]["dur"] = duration
                durations.append(duration)
            
            # Calculate average duration if there are any events
            func_time = sum(durations) / len(durations) if durations else 0
            
            return func_dict, func_time

        cublas_dict, cublas_time = process_events(data, "Cublas_record")
        triton_dict, triton_time = process_events(data, "Triton_record")
        print(f"    Pure kernel Elapsed time {round((triton_time), 2)} ms, {round((2*M*n*k )/(triton_time)/10**9, 0)} tflops")
        print(f"    Cublas kernel Elapsed time {round((cublas_time), 2)} ms, {round((2*M*n*k )/(cublas_time)/10**9, 0)} tflops, acceleration {cublas_time/triton_time: .2f}")