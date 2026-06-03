import torch
from torch.profiler import ProfilerActivity, profile, record_function
import os
import sys
from pathlib import Path

_REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "gemm").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gemm.FP8.k_grouped_gemm_channel_expert import matmul

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
