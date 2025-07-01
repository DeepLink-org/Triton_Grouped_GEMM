import torch
from torch.profiler import ProfilerActivity, profile, record_function
import os
import sys
from pathlib import Path

from gemm.BF16.m_grouped_gemm_masked_TMA import m_grouped_gemm_masked
from gemm.BF16.utils import generate_random_list, row_max_normalization

if __name__=='__main__':
    from typing import Tuple
    import random
    
    from torch.profiler import ProfilerActivity, profile, record_function
    # import grouped_gemm_backend as backend

    from gemm.BF16.utils import generate_random_list, row_max_normalization

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
    trans_b = False; print(f"{trans_b = }")
    expected_m = 4096
    device = f"cuda:{torch.cuda.device_count()-1}"
    batch_sizes = expected_m * torch.ones(groups, device = device, dtype = torch.int32)

    masked_m = torch.Tensor(generate_random_list(groups, groups*1024)).to(device).to(torch.int64).abs()
    masked_m[0:127:5] = 0
    # masked_m = batch_sizes
    masked_m = torch.where(masked_m > batch_sizes, batch_sizes, masked_m)
    # print(f"{masked_m = }")
    
    batch_sizes_cpu = batch_sizes.cpu()
    M = batch_sizes.sum().item()
    M_masked = masked_m.sum().item()

    for (n, k) in ((768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)):
        torch.cuda.empty_cache()
        a = torch.randn(M, k, dtype = torch.bfloat16, device = device).view(-1, k).requires_grad_(True)
        b = torch.randn(z, n, k, dtype = torch.bfloat16, device = device) if trans_b else torch.randn(z, k, n, dtype = torch.bfloat16, device = device).requires_grad_(True)
        out_ref = gmm(a, b, batch_sizes.cpu(), trans_b)
        out_triton = torch.empty((M, n), dtype = torch.bfloat16, device = device)

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
                m_grouped_gemm_masked(a, b, out_triton, masked_m, expected_m, trans_b)
                torch.cuda.synchronize(device = device)
                prof.step()

        # post-process, row normalization
        out_triton = row_max_normalization(out_triton)
        out_ref = row_max_normalization(out_ref)

        group_end = batch_sizes.cumsum(0) - batch_sizes + masked_m
        group_start = batch_sizes.cumsum(0) - batch_sizes
        for g in range(groups):
            torch.testing.assert_close(out_triton[group_start[g]:group_end[g], :], out_ref[group_start[g]:group_end[g], :], rtol = 0.001, atol = 0.005)
        print(f"{n = }, {k = }, {M_masked = }")
        

        import json
        with open(trace_file, "r") as file:
            data = json.load(file)

        kernel_time = 1000
        for event in data["traceEvents"]:
            try:
                if "m_grouped_gemm_masked_kernel" in event["name"]:
                    kernel_time = min(event["dur"] / 1000, kernel_time)
            except:
                pass
        print(f"    Pure kernel Elapsed time {round((kernel_time), 2)} ms, {round((2*M_masked*n*k )/(kernel_time)/10**9, 0)} tflops")

