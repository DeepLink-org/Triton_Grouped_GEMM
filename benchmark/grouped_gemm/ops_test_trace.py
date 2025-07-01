# Copyright (c) 2025, DeepLink.
import unittest
import itertools
from absl.testing import parameterized
from grouped_gemm import ops
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from pathlib import Path
import json
import os

def allclose(x, y, pct=2.0):
    mask = torch.isclose(x, y, rtol=1e-5)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True

def add_transpose_flags(x):
    out = []
    for y in x:
        for f in [(False,), (True,)]:
            out.append(y + f)
    return out

_TEST_PROBLEMS = add_transpose_flags((
    (128, 5120, 1536, 4096),
    (128, 5120, 2048, 1536),
    (128, 5120, 768, 2048),
    # (128, 5120, 4096, 3072),
))

def randn(bs, x, y):
    out = (torch.rand(bs, x, y) - 0.5 * 2) / (y * x)
    return out.cuda().to(torch.bfloat16)

def gmm(a, b, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.numpy()
    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)



@parameterized.parameters(*_TEST_PROBLEMS)
class OpsTest(parameterized.TestCase):

    # def testGroupedGemm_FixedSizes(self, z, m, k, n, trans_b):
    #     torch.manual_seed(0)
    #     a = randn(z, m, k).view(-1, k)
    #     b = randn(z, n, k) if trans_b else randn(z, k, n)
    #     batch_sizes = torch.tensor([m] * z)

    #     a.requires_grad_(True)
    #     b.requires_grad_(True)
    #     a_ref = a.detach().clone().requires_grad_(True)
    #     b_ref = b.detach().clone().requires_grad_(True)

    #     out = ops.gmm(a, b, batch_sizes, trans_b)
    #     expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
    #     self.assertTrue(allclose(out, expected_out))

    #     # Check gradients.
    #     out.sum().backward()
    #     expected_out.sum().backward()
    #     self.assertTrue(allclose(a.grad, a_ref.grad))
    #     self.assertTrue(allclose(b.grad, b_ref.grad))

    # def testGroupedGemm_VariableSizes(self, z, m, k, n, trans_b):
    #     torch.manual_seed(0)
    #     a = randn(z, m, k).view(-1, k)
    #     b = randn(z, n, k) if trans_b else randn(z, k, n)

    #     dist = torch.rand(z, )
    #     dist /= dist.sum()
    #     batch_sizes = (dist * m).to(torch.long)
    #     error = m * z - batch_sizes.sum()
    #     batch_sizes[-1] += error
    #     self.assertEqual(batch_sizes.sum(), m * z)

    #     a.requires_grad_(True)
    #     b.requires_grad_(True)
    #     a_ref = a.detach().clone().requires_grad_(True)
    #     b_ref = b.detach().clone().requires_grad_(True)

    #     out = ops.gmm(a, b, batch_sizes, trans_b)
    #     expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
    #     self.assertTrue(allclose(out, expected_out))

    #     # Check gradients.
    #     out.sum().backward()
    #     expected_out.sum().backward()
    #     self.assertTrue(allclose(a.grad, a_ref.grad))
    #     self.assertTrue(allclose(b.grad, b_ref.grad))

    # def testGroupedGemm_FixedSizes_Profile(self, z, m, k, n, trans_b):
    #     torch.manual_seed(0)
    #     a = randn(z, m, k).view(-1, k)
    #     b = randn(z, n, k) if trans_b else randn(z, k, n)
    #     batch_sizes = torch.tensor([m] * z)

    #     script_path = Path(__file__).resolve()
    #     parent_dir = script_path.parent
    #     trace_file = f"{parent_dir}/trace/gmm_fixed_sizes_z{z}_m{m}_k{k}_n{n}_transb{trans_b}.json"
    #     Path(os.path.join(parent_dir, "trace")).mkdir(parents=True, exist_ok=True)
    #     active_ = 3

    #     def trace_handler(prof):
    #         prof.export_chrome_trace(trace_file)

    #     with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         schedule=torch.profiler.schedule(wait=1, warmup=3, active=active_, repeat=0),
    #         on_trace_ready=trace_handler,
    #         with_modules=True,
    #         record_shapes=True,
    #     ) as prof:
    #         for i in range(4 + active_):
    #             with record_function("gmm_record"):
    #                 out = ops.gmm(a, b, batch_sizes, trans_b)
    #             torch.cuda.synchronize()
    #             prof.step()

    #     # Verify correctness
    #     expected_out = gmm(a, b, batch_sizes, trans_b)
    #     self.assertTrue(allclose(out, expected_out))

    #     # Calculate TFLOPS
    #     with open(trace_file, "r") as file:
    #         data = json.load(file)
    #     kernel_time = 0
    #     for event in data["traceEvents"]:
    #         try:
    #             if "gmm" in event["name"].lower():
    #                 kernel_time += event["dur"] / 1000  # Convert to ms
    #         except:
    #             pass
    #     kernel_time /= active_
    #     total_flops = 2 * z * m * n * k  # FLOPs for matrix multiplication
    #     tflops = total_flops / kernel_time / 1e9  # TFLOPS
    #     print(f"FixedSizes (z={z}, m={m}, k={k}, n={n}, trans_b={trans_b}): "
    #           f"Kernel time {round(kernel_time, 2)} ms, {round(tflops, 0)} TFLOPS")

    def testGroupedGemm_VariableSizes_Profile(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, n, k) if trans_b else randn(z, k, n)

        dist = torch.rand(z, )
        dist /= dist.sum()
        batch_sizes = (dist * m).to(torch.long)
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        self.assertEqual(batch_sizes.sum(), m * z)

        script_path = Path(__file__).resolve()
        parent_dir = script_path.parent
        trace_file = f"{parent_dir}/trace/gmm_variable_sizes_z{z}_m{m}_k{k}_n{n}_transb{trans_b}.json"
        Path(os.path.join(parent_dir, "trace")).mkdir(parents=True, exist_ok=True)
        active_ = 3

        def trace_handler(prof):
            prof.export_chrome_trace(trace_file)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=active_, repeat=0),
            on_trace_ready=trace_handler,
            with_modules=True,
            record_shapes=True,
        ) as prof:
            for i in range(4 + active_):
                out = ops.gmm(a, b, batch_sizes, trans_b)
                torch.cuda.synchronize()
                prof.step()

        # Verify correctness
        expected_out = gmm(a, b, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

        # Calculate TFLOPS
        with open(trace_file, "r") as file:
            data = json.load(file)
        kernel_time = 0
        for event in data["traceEvents"]:
            try:
                if "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas" in event["name"] \
                    or "sm90_xmma_gemm_bf16bf16_bf16f32_f32_nn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas" in event["name"]:
                    kernel_time += event["dur"] / 1000  # Convert to ms
            except:
                pass
        kernel_time /= active_
        total_flops = 2 * batch_sizes.sum().item() * n * k  # Convert sum to scalar
        tflops = total_flops / kernel_time / 1e9  # TFLOPS
        print(f"VariableSizes (z={z}, m={m}, k={k}, n={n}, trans_b={trans_b}): "
              f"Kernel time {round(kernel_time, 2)} ms, {round(tflops, 0)} TFLOPS")

if __name__ == '__main__':
    unittest.main()