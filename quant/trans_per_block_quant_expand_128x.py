# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torch.profiler import ProfilerActivity, profile

@triton.jit
def get_group_id(m, group_offsets, g_start, num_experts):
    id = 0
    off_out = 0
    offnxt_out = 0
    for group_id in tl.range(g_start, num_experts):
        group_off = tl.load(group_offsets + group_id)
        group_off_nxt = tl.load(group_offsets + group_id  + 1)
        if m >=  group_off and m < group_off_nxt:
            id = group_id
            off_out = group_off
            offnxt_out = group_off_nxt
    return id, off_out, offnxt_out

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
def trans_per_block_quant_expand_128x_kernel(
    input_ptr,
    output_ptr,
    output_scales_ptr,
    group_pad_offs,
    token_cumdiffs,
    token_ends,
    num_experts: tl.constexpr,
    M_pad_ptr,
    M,
    N: tl.constexpr,
    M_EXPAND,
    fmax: tl.constexpr,
    fmin: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Dequant+transpose+quant kernel."""

    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_tiles_pad = tl.cdiv(M_EXPAND, BLOCK_M) * num_pid_n

    for tile_id in tl.range(start_pid+num_tiles, num_tiles_pad, BLOCKS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_an = pid_n * BLOCK_M + tl.arange(0, BLOCK_N)
        offs_bm = offs_an
        offs_bn = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        output_offset = (
            output_ptr
            + offs_bn[None, :]
            + offs_bm[:, None] * M_EXPAND
        )
        output_scale_offset = (
            output_scales_ptr
            + pid_n * (M_EXPAND // 128) + pid_m
        )

        out = tl.zeros((BLOCK_M, BLOCK_M), dtype = output_ptr.dtype.element_ty)
        tl.store(output_offset, out)
        tl.store(output_scale_offset, 0)


    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        group, _, _ = get_group_id(pid_m * BLOCK_M, group_pad_offs, 0, num_experts)
        token_cumdiff = tl.load(token_cumdiffs + group)
        token_end = tl.load(token_ends + group)

        offs_am = (pid_m * BLOCK_M - token_cumdiff + tl.arange(0, BLOCK_M))
        offs_an = pid_n * BLOCK_M + tl.arange(0, BLOCK_N)
        offs_bm = offs_an
        offs_bn = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        input_offset = (
            input_ptr
            + offs_am[:, None] * N
            + offs_an[None, :] * 1
        )
        output_offset = (
            output_ptr
            + offs_bn[None, :]
            + offs_bm[:, None] * M_EXPAND
        )
        output_scale_offset = (
            output_scales_ptr
            + pid_n * (M_EXPAND // 128) + pid_m
        )

        mask_input = (offs_am[:, None] < token_end) & (offs_an < N)
        input_block = tl.load(
            input_offset, mask=mask_input, other=0.0
        )

        output_block_scale = tl.max(tl.max(tl.abs(input_block), 0), 0) / fmax
        output_block_scale = tl.clamp(output_block_scale, 1e-12, 3e38)
        input_block = input_block / output_block_scale
        input_block = tl.clamp(input_block, fmin, fmax).trans(1,0).to(output_ptr.dtype.element_ty)

        mask_out = (offs_bm[None, :] < N)
        tl.store(output_offset, input_block, mask = mask_out)
        tl.store(output_scale_offset, output_block_scale)


# @triton_op("myfp8::trans_per_block_quant_expand_128x", mutates_args={})
def trans_per_block_quant_expand_128x(
    input_tensor: torch.Tensor,
    size_per_group: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequant a per-group per-tile quantized tensor. Do per-block quantization
    along another dimension. And finally transpose it.

    Args:
        input_tensor (Tensor): The input quantized tensor. Shape [M, N]. It is
            per-tile quantized along N dim.
        input_scales (Tensor): The input scale. Shape [M, group_num_n].
            group_num_n=N//group_size.
        size_per_group (Tensor): The seq length of each expert. The sum of it
            should be equal to M.
        group_size (int): The group size of the quantization. Default to 128.

    Returns:
        output_tensor (Tensor): The output tensor. Shape [N, M]. It is per-tile
            quantized along M dim.
        output_scales (Tensor): The output scales. Shape [N, group_num_m].
            group_num_m=(M_EXPAND+group_size-1)//group_size.
    """
    M, N = input_tensor.shape # (302873, 7168)
    assert N % group_size == 0
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    group_pad_off = torch.zeros(size_per_group.shape[0] + 1, device = "cuda", dtype = torch.int32)
    
    BLOCK_M = group_size
    size_per_group_padding = triton.cdiv(size_per_group, BLOCK_M) * BLOCK_M
    group_pad_off[1:] = size_per_group_padding.cumsum(0)

    num_experts = size_per_group.shape[0]
    M_EXPAND = M + group_size * num_experts - M % group_size
    output_tensor = input_tensor.new_empty((N, M_EXPAND), dtype=dtype)
    output_scales = input_tensor.new_empty(
        (N // group_size, triton.cdiv(M_EXPAND, group_size)), dtype=torch.float32
    )

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (NUM_SMS, )

    M_pad = size_per_group_padding.sum()

    token_diff = size_per_group_padding - size_per_group
    token_cumdiff = token_diff.cumsum(0)
    token_pad_end = size_per_group_padding.cumsum(0) - token_cumdiff
    token_end = size_per_group.cumsum(0) 

    token_cumdiff = token_diff.cumsum(0) - token_diff

    wrap_triton(trans_per_block_quant_expand_128x_kernel)[grid](
        input_tensor,
        output_tensor,
        output_scales,
        group_pad_off,
        token_cumdiff,
        token_end,
        num_experts,
        M_pad,
        M,
        N,
        M_EXPAND,
        fmax=fmax,
        fmin=fmin,
        BLOCK_M=group_size,
        BLOCK_N=group_size,
    )

    return output_tensor, output_scales, size_per_group_padding


# @torch.compile(fullgraph=True)
def trans_quant(input_tensor, size_per_group):
    return trans_per_block_quant_expand_128x(input_tensor, size_per_group)

if __name__ == "__main__":
    def to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype):
        max_value = torch.finfo(float8_dtype).max
        x = x.clamp(min=-max_value, max=max_value)
        return x.to(float8_dtype)


    def per_block_trans_quant_expand_per_expert(
        x, eps=1e-12, block_size=128, quant_dtype=torch.float8_e4m3fn
    ):
        x = x.T
        dim, seq = x.shape
        seq_expand = triton.cdiv(seq, block_size) * block_size
        x_expand = torch.cat([x, x.new_zeros((dim, seq_expand - seq))], dim=-1)

        x_expand = (
            x_expand.reshape(
                dim // block_size, block_size, seq_expand // block_size, block_size
            )
            .transpose(1, 2)
            .reshape(-1, block_size * block_size)
        )
        amax = x_expand.abs().amax(-1, True)
        scale = amax.float() / torch.finfo(quant_dtype).max
        x_fp8 = to_fp8_saturated(x_expand / scale, quant_dtype)
        x_fp8 = (
            x_fp8.reshape(
                dim // block_size, seq_expand // block_size, block_size, block_size
            )
            .transpose(1, 2)
            .reshape(dim, seq_expand)
        )
        scale = scale.reshape(dim // block_size, seq_expand // block_size)

        return x_fp8, scale


    def per_block_trans_quant_expand_torch(x, tokens_per_expert):
        M = x.shape[0]
        ne = tokens_per_expert.shape[0]

        x_list = torch.split(x, tokens_per_expert.tolist(), dim=0)
        x_trans_quant_list, x_trans_quant_scale_list = [], []
        for x in x_list:
            x_fp8, scale = per_block_trans_quant_expand_per_expert(x)
            x_trans_quant_list.append(x_fp8)
            x_trans_quant_scale_list.append(scale)
        x_trans_quant = torch.cat(x_trans_quant_list, dim=-1)
        x_trans_quant_scale = torch.cat(x_trans_quant_scale_list, dim=-1)

        pad_len = M + 128 * ne - M % 128 - x_trans_quant.shape[1]
        pad = x_trans_quant.new_zeros((x_trans_quant.shape[0], pad_len))
        x_trans_quant = torch.cat([x_trans_quant, pad], dim=1)

        pad_len = (M + 128 * ne - M % 128) // 128 - x_trans_quant_scale.shape[1]
        pad = x_trans_quant_scale.new_zeros((x_trans_quant_scale.shape[0], pad_len))
        x_trans_quant_scale = torch.cat([x_trans_quant_scale, pad], dim=1)

        return x_trans_quant, x_trans_quant_scale, triton.cdiv(tokens_per_expert, 128) * 128


    input_tensor = torch.randn(65536, 7168, device='cuda', dtype=torch.bfloat16)
    size_per_group = torch.tensor([65536 // 32] * 32).cuda()

    for _ in range(3):
        output_tensor, output_scales, size_per_group_expand = trans_quant(input_tensor, size_per_group)
        output_tensor_ref, output_scales_ref, size_per_group_expand_ref = per_block_trans_quant_expand_torch(input_tensor, size_per_group)
    out = (output_tensor_ref.to(torch.float) - output_tensor.to(torch.float))
    scale = (output_scales_ref - output_scales)
    print(f"max diff with torch out = {out.abs().max()}, scale = {scale.abs().max()}")

    # with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         # with_stack = True,
    #         # with_modules = True,
    #         record_shapes=True,
    #     ) as prof:
    #     for _ in range(3):
    #         output_tensor, output_scales, size_per_group_expand = trans_quant(input_tensor, size_per_group)
    #         output_tensor, output_scales, size_per_group_expand = trans_quant(input_tensor2, size_per_group)

    # prof.export_chrome_trace('trans_per_block_quant_expand_128x.json')
