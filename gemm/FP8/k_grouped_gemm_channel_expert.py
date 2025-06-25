
import torch
import numpy as np
import triton
import triton.language as tl


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, "GROUP_M": 8}, num_stages=3,
                      num_warps=8,),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, "GROUP_M": 3}, num_stages=3,
                      num_warps=8,),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, "GROUP_M": 16}, num_stages=3,
                      num_warps=8,),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, "GROUP_M": 8}, num_stages=5,
                      num_warps=8,),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, "GROUP_M": 3}, num_stages=5,
                      num_warps=8,),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, "GROUP_M": 8}, num_stages=7,
        #               num_warps=8,),
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

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n

@triton.jit
def grouped_launch(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n

@triton.autotune(configs=get_cuda_autotune_config(), key=['M','N'])
@triton.jit
def gemm_kernel_tma(
                    a_desc_ptr, b_desc_ptr, c_desc_ptr,
                    a_scale, b_scale,
                    tokens_per_expert,
                    tokens_off,
                    num_groups,
                    M, N, K,
                    dtype_a: tl.constexpr, dtype_b: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, 
                    GROUP_M: tl.constexpr,
                    NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n * num_groups

    dtypeA = tl.float8e4nv if dtype_a == 1  else tl.float8e5
    dtypeB = tl.float8e4nv if dtype_b == 1  else tl.float8e5

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        g = tile_id // (num_pid_m * num_pid_n)

        id_tmp = tile_id % (num_pid_m * num_pid_n)

        if GROUP_M == 1:
            num_pid_m = tl.cdiv(M, BLOCK_M)
            pid_m = id_tmp % num_pid_m
            pid_n = id_tmp // num_pid_m
        else:
            pid_m, pid_n = grouped_launch(id_tmp, M, N, BLOCK_M, BLOCK_N, GROUP_M)

        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        offs_k = tl.load(tokens_off + g)

        offs_as = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) + g * M
        offs_bs = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) + g * N
        as_ptrs = a_scale + offs_as[:, None]
        bs_ptrs = b_scale + offs_bs[:, None]

        tokens = tl.load(tokens_per_expert + g)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        num_pid_k = tl.cdiv(tokens, BLOCK_K)

        a_s = tl.load(as_ptrs)
        b_s = tl.load(bs_ptrs)

        for kk in range(0, num_pid_k):
            a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], dtypeA)
            b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K], dtypeB)
            accumulator = tl.dot(a, b.T, acc=accumulator, out_dtype=tl.float32)
            offs_k += BLOCK_K
        accumulator *= (a_s * b_s.T)
        accumulator = accumulator.to(tl.bfloat16)
        tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am + g * M, offs_bn])


HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)
# TmaAutoTuneHelper used in htyu's PR #5622
class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:

        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 512

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_1d_tma_descriptor)
        self.fill_2d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_2d_tma_descriptor)
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8)

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]

    
class KernelParamWrapper:
    def __init__(self, desc):
        self.desc = desc

    def tma_desc_cpu_ptr(self):
        return self.desc.data_ptr()


def matmul(
        x_fp8: torch.Tensor,  # (M, K)
        x_scale: torch.Tensor, # (M, ne)
        y_fp8: torch.Tensor,   # (N, K)
        y_scale: torch.Tensor,   # (N, ne)
        tokens_per_expert: torch.Tensor,   # (ne, )
        M: int, 
        N: int, 
        K: int, 
        num_groups: int, 
        dtype_a: int = 1, 
        dtype_b: int = 1) -> torch.Tensor:
    desc_helper = TmaAutoTuneHelper()

    output = torch.empty((num_groups, M, N), dtype=torch.bfloat16, device='cuda')
    token_off = tokens_per_expert.cumsum(0) - tokens_per_expert
    token_off = token_off.int()

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            x_fp8.data_ptr(),
            M,
            K,
            META["BLOCK_M"],
            META["BLOCK_K"],
            x_fp8.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            y_fp8.data_ptr(),
            N,
            K,
            META["BLOCK_N"],
            META["BLOCK_K"],
            y_fp8.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "c",
            output.data_ptr(),
            num_groups * M,
            N,
            META["BLOCK_M"],
            META["BLOCK_N"],
            output.element_size(),
        )
        # return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
        return (
            (NUM_SMS,)
            )

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")
    gemm_kernel_tma[grid](
        desc_a, desc_b, desc_c,
        x_scale,
        y_scale,
        tokens_per_expert, 
        token_off,
        num_groups,
        M, N, K,
        dtype_a, dtype_b,
        NUM_SMS=NUM_SMS
    )

    # print(f"best config {gemm_kernel_tma.best_config}", flush = True)
    return output

if __name__ == "__main__":
    from typing import Tuple
    def cell_div(x: int, y: int) -> int:
        """
        Perform ceiling division of two integers.

        Args:
            x: the dividend.
            y: the divisor.

        Returns:
            The result of the ceiling division.
        """
        return (x + y - 1) // y

    def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

    def per_channel_cast_to_fp8(x: torch.Tensor, dtype = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, n)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        if dtype == 1:
            fmax = torch.finfo(torch.float8_e4m3fn).max
            return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / fmax).view(m, -1)
        else:
            fmax = torch.finfo(torch.float8_e5m2).max
            return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e5m2).view(m, n), (x_amax / fmax).view(m, -1)


    def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2
        m, n = x.shape
        x_padded = torch.zeros((cell_div(m, 128) * 128, cell_div(n, 128) * 128), dtype=x.dtype, device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

    def per_expert_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2
        m, n = x.shape
        x_padded = torch.zeros((cell_div(m, 128) * 128, cell_div(n, 128) * 128), dtype=x.dtype, device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(1, m, 1, n)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

    def gen_data(M, N, tokens_per_expert, ref_dw, dtype_a = 1, dtype_b = 1):
        x_fp8_list, x_scale_list, y_fp8_list, y_scale_list = [], [], [], []
        tokens_per_expert_pad = []
        # prepare data
        for i, tokens in enumerate(tokens_per_expert):
            tokens = int(tokens)
            tokens_padding = int((tokens + 127) / 128) * 128
            tokens_per_expert_pad.append(tokens_padding)
            x = torch.randn(M, tokens, device='cuda', dtype=torch.bfloat16)
            y = torch.randn(N, tokens, device='cuda', dtype=torch.bfloat16)

            ref_dw[i,:,:] = (x @ y.T)

            x_padding = torch.zeros(M, tokens_padding, device='cuda', dtype=torch.bfloat16)
            y_padding = torch.zeros(N, tokens_padding, device='cuda', dtype=torch.bfloat16)
            x_padding[:,:tokens] = x
            y_padding[:,:tokens] = y

            x_fp8, x_scale = per_channel_cast_to_fp8(x_padding, dtype_a)
            y_fp8, y_scale = per_channel_cast_to_fp8(y_padding, dtype_b)

            x_fp8_list.append(x_fp8); x_scale_list.append(x_scale)
            y_fp8_list.append(y_fp8); y_scale_list.append(y_scale)

        # input 都按照 token 连续，scale 都 按照 token stride
        x_fp8 = torch.cat(x_fp8_list, -1).to(dtype=x_fp8.dtype).contiguous()
        x_scale = torch.cat(x_scale_list, -1).contiguous().transpose(0, 1).contiguous().transpose(0, 1)
        y_fp8 = torch.cat(y_fp8_list, -1).to(dtype=y_fp8.dtype).contiguous()
        y_scale = torch.cat(y_scale_list, -1).contiguous().transpose(0, 1).contiguous().transpose(0, 1)

        return x_fp8, x_scale, y_fp8, y_scale, tokens_per_expert_pad

        
    bias = 3
    # tokens_per_expert = [65536] * 1
    # tokens_per_expert  = [2048] * 4
    # tokens_per_expert  = [1] * 31 + [65536 - 31]
    # tokens_per_expert  = [1, 1, 1, 1, 1, 1, 1, 65536 - 7]
    tokens_per_expert  = [512 - bias, 2 * 2048 - 512 + bias, 128 - bias, 2 * 2048 - 128 + bias] * 8
    # tokens_per_expert  = [2 * 1024 - bias, 8 * 1024 + bias, 14 * 1024 - bias, 7 * 1024 + bias, 8 * 1024 - bias, 9 * 1024 + bias, 7.5 * 1024+bias, 8.5 * 1024 - bias]
    num_groups = len(tokens_per_expert)
    
    # type 1: e4m3; type2: e5m2
    dtype_a = 2
    dtype_b = 1
    
    # First layer of MLP
    M1 = 5120; N1 = 6144
    # M1 = 128; N1 = 128
    ref_dw1 = torch.zeros((num_groups, M1, N1), device='cuda', dtype=torch.bfloat16)
    x_fp81, x_scale1, y_fp81, y_scale1, tokens_per_expert_pad = gen_data(M1, N1, tokens_per_expert, ref_dw1, dtype_a, dtype_b)
    # Second layer of MLP
    M2 = 3072; N2 = 5120
    # M2 = 5120; N2 = 6144
    ref_dw2 = torch.zeros((num_groups, M2, N2), device='cuda', dtype=torch.bfloat16)
    x_fp82, x_scale2, y_fp82, y_scale2, tokens_per_expert_pad = gen_data(M2, N2, tokens_per_expert, ref_dw2, dtype_a, dtype_b)

    tokens_per_expert_pad= torch.Tensor(tokens_per_expert_pad).to("cuda").to(torch.int32)
    K = int(sum(tokens_per_expert_pad))

    from pathlib import Path
    script_path = Path(__file__).resolve()
    parent_dir = script_path.parent.parent
    trace_file = f"{parent_dir}/trace/k_grouped_gemm_fp8_act_per_channel_w_per_expert"  + ".json"
    import os
    Path(os.path.join(parent_dir, "trace")) .mkdir(parents=True, exist_ok=True)
    active_ = 3
    def trace_handler(prof):
        prof.export_chrome_trace(trace_file)
    
    from torch.profiler import ProfilerActivity, profile, record_function
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

            output1 = matmul(x_fp81, x_scale1, y_fp81, y_scale1, tokens_per_expert_pad, M1, N1, K, num_groups, dtype_a, dtype_b)
            # output2 = matmul(x_fp82, x_scale2, y_fp82, y_scale2, tokens_per_expert_pad, M2, N2, K, num_groups, dtype_a, dtype_b)
            torch.cuda.synchronize()
            prof.step()

    amax = max(output1.abs().max(), ref_dw1.abs().max())
    adiffmax = (output1 - ref_dw1).abs().max()
    rdiffmax = adiffmax / amax
    print(f"max relative difference of the first layer is {rdiffmax}")

    # amax = max(output2.abs().max(), ref_dw2.abs().max())
    # adiffmax = (output2 - ref_dw2).abs().max()
    # rdiffmax = adiffmax / amax
    # print(f"max relative difference of the second layer is {rdiffmax}")
    
    # Get time from trace
    import json
    with open(trace_file, "r") as file:
        data = json.load(file)
    
    kernel_time = 0
    for event in data["traceEvents"]:
        if "gemm_kernel_tma" in event["name"]:
            kernel_time += event["dur"] / 1000
    print(f"\nPure kernel Elapsed time {round((kernel_time), 1)} ms, {round(2*K*(N1*M1 + N2*M2)/(kernel_time)/10**9, 0)} tflops")