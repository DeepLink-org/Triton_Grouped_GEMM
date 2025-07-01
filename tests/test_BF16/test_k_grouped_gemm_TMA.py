import torch
from torch.profiler import ProfilerActivity, profile, record_function
import os
import sys
from pathlib import Path

from gemm.BF16.k_grouped_gemm_TMA import k_grouped_gemm
from gemm.BF16.utils import generate_random_list, row_max_normalization

if __name__=='__main__':
    from typing import Tuple
    import random
    
    from torch.profiler import ProfilerActivity, profile, record_function
    from torch.library import triton_op, wrap_triton

    from gemm.BF16.utils import generate_random_list, row_max_normalization

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
    K = batch_sizes.sum().item()

    for (m, n) in ((768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)):
        torch.cuda.empty_cache()
        a = torch.randn(K, m, dtype = torch.bfloat16, device = "cuda").view(K, -1)
        b = torch.randn(K, n, dtype = torch.bfloat16, device = "cuda").view(K, -1)
        out_ref = gmm_dw(a, b, batch_sizes.cpu())
        out_cublas = torch.empty_like(out_ref)
        out_cutlass = torch.empty_like(out_ref)

        from pathlib import Path
        script_path = Path(__file__).resolve()
        parent_dir = script_path.parent.parent
        trace_file = f"{parent_dir}/trace/gmm_dw_triton_cublas_cutlass_M{m}_N{n}"  + ".json"
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
                    out_triton = k_grouped_gemm(a, b, batch_sizes)
                torch.cuda.synchronize()
                prof.step()

        # post-process, row normalization
        out_triton = row_max_normalization(out_triton)
        out_ref = row_max_normalization(out_ref)

        torch.testing.assert_close(out_triton, out_ref, rtol = 0.001, atol = 0.01)

        print(f"{m = }, {n = }, {K = }")
        

        import json
        with open(trace_file, "r") as file:
            data = json.load(file)

        triton_time = 0
        cublas_time = 0
        cutlass_time = 0
        for event in data["traceEvents"]:
            try:
                if "k_grouped_gemm" in event["name"]:
                    triton_time += event["dur"] / 1000
            except:
                pass
        triton_time /= active_
        print(f"    Pure Triton kernel Elapsed time {round((triton_time), 2)} ms, {round((2*m*n*K )/(triton_time)/10**9, 0)} tflops")