import torch
from torch.profiler import ProfilerActivity, profile, record_function
import os
import sys
from pathlib import Path
from gemm.FP8.m_grouped_gemm_channel_expert import gmm_fp8_act_per_channel_w_per_expert


if __name__=='__main__':
    from typing import Tuple

    def ceil_div(x: int, y: int) -> int:
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

    def per_channel_cast_to_fp8(x: torch.Tensor, dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, n)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        if dtype == torch.float8_e4m3fn:
            fmax = torch.finfo(torch.float8_e4m3fn).max
            return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / fmax).view(m, -1)
        else:
            fmax = torch.finfo(torch.float8_e5m2).max
            return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e5m2).view(m, n), (x_amax / fmax).view(m, -1)


    def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2
        m, n = x.shape
        x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

    def per_expert_cast_to_fp8(x: torch.Tensor, dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 3
        num_groups, m, n = x.shape
        x_padded = torch.zeros((num_groups, ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
        x_padded[:, :m, :n] = x
        x_view = x_padded.view(num_groups, m, 1, n)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        if dtype == torch.float8_e4m3fn:
            fmax = torch.finfo(torch.float8_e4m3fn).max
            x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e4m3fn)
            return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (x_amax / fmax).view(x_view.size(0), x_view.size(2))
        else:
            fmax = torch.finfo(torch.float8_e5m2).max
            x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e5m2)
            return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (x_amax / fmax).view(x_view.size(0), x_view.size(2))


    def gen_data_fwd(M, N, K, tokens_per_expert, dtype_out = torch.bfloat16, dtype_a = torch.float8_e4m3fn, dtype_b = torch.float8_e4m3fn):
        ref_dw = torch.empty(M, N, device = "cuda", dtype = dtype_out)
        x = torch.randn(M, K, device = "cuda", dtype = torch.bfloat16)

        num_groups = len(tokens_per_expert)

        weights = torch.randn(num_groups, N, K, device = "cuda", dtype = torch.bfloat16)

        x_fp8, x_scale = per_channel_cast_to_fp8(x, dtype_a)
        weights_fp8, weights_scale = per_expert_cast_to_fp8(weights, dtype_b)

        # prepare data
        t_start = 0
        for i, tokens in enumerate(tokens_per_expert):
            tokens = int(tokens)
            x_tmp = x[t_start: t_start+tokens]; 
            weight = weights[i]

            ref_dw[t_start: t_start+tokens] = (x_tmp @ weight.T)

            t_start += tokens

        # breakpoint()
        return x_fp8, x_scale, weights_fp8, weights_scale, ref_dw

    # Example usage
    # dtype_a = torch.float8_e4m3fn
    # dtype_b = torch.float8_e4m3fn

    dtype_a = torch.float8_e5m2
    dtype_b = torch.float8_e4m3fn

    dtype_out = torch.bfloat16

    # from helper import gen_data_fwd

    bias = 3
    # tokens_per_expert = [2047, 2048] * 1
    tokens_per_expert = [512 - bias, 2 * 2048 - 512 + bias, 128 - bias, 2 * 2048 - 128 + bias] * 8
    # tokens_per_expert  = [1] * 31 + [65536 - 31]
    M = sum(tokens_per_expert)
    N = 6144
    K = 5120

    # N = 5120
    # K = 3072

    x_fp8, x_scale, weights_fp8, weights_scale, ref_fwd = gen_data_fwd(M, N, K, tokens_per_expert, dtype_out = dtype_out, dtype_a = dtype_a, dtype_b = dtype_b)
    size_per_group = torch.tensor(tokens_per_expert, device='cuda', dtype=torch.int)
    weights_scale = weights_scale.view(-1, 1).repeat(1, N)
    
    
    from pathlib import Path
    script_path = Path(__file__).resolve()
    parent_dir = script_path.parent.parent
    trace_file = f"{parent_dir}/trace/m_grouped_gemm_fp8_act_per_channel_w_per_expert"  + ".json"
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
            output_tensor = gmm_fp8_act_per_channel_w_per_expert(x_fp8, x_scale, weights_fp8, weights_scale, size_per_group, dtype_out = dtype_out)

            torch.cuda.synchronize()
            prof.step()

    amax = max(output_tensor.abs().max(), ref_fwd.abs().max())
    adiffmax = (output_tensor - ref_fwd).abs().max()
    rdiffmax = adiffmax / amax
    # import pdb; pdb.set_trace()
    print(f"max relative difference of the layer is {rdiffmax}")
    
    # Get time from trace
    import json
    with open(trace_file, "r") as file:
        data = json.load(file)
    
    kernel_time = 0
    for event in data["traceEvents"]:
        if "gmm_fp8_act_per_channel_w_per_expert_kernel" in event["name"]:
            kernel_time += event["dur"] / 1000
    print(f"\nPure kernel Elapsed time {round((kernel_time), 1)} ms, {round((2* M * N *K)/(kernel_time)/10**9, 0)} tflops")

    
