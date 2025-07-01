# Grouped GEMM Implementation using Triton
This repository provides optimized implementations of grouped GEMM using Triton. The implementations are categorized by data type: bfloat16 and FP8. Each implementation includes unit test that can be run directly from the python scripts.

## Features
- Optimized grouped GEMM implementations for BF16 and FP8 data types
- Support for different grouping strategies (M-Grouped and K_grouped)
- Various quantization schemes for FP8 implementations
- Included unit tests for varification

## BF16 Implementations

The following BF16 implementations are available in `gemm/BF16`:


| Script | Gemm Type |
|---------|---------|
|`m_grouped_gemm_masked_TMA.py` | M-Grouped Gemm with mask | 
|`m_grouped_gemm_TMA.py` | M-Grouped Gemm |
|`k_grouped_gemm_TMA.py` | K-Grouped Gemm |


M-Grouped Gemm with TMA 
| trans_b | Dimensions (M x N x K,z=128) | ours_Elapsed Time (ms) | ours_TFLOPS | Benchmark_Elapsed Time (ms) | Benchmark_TFLOPS |
|:-------:|:----------------------------:|:----------------------:|:-----------:|:---------------------------:|:----------------:|
| True    | 655360 x 2048 x 768          | 3.53                   | 584.0       | 2.55                        | 808.0            |
| True    | 655360 x 1536 x 2048         | 5.79                   | 712.0       | 4.90                        | 842.0            |
| True    | 655360 x 4096 x 1536         | 11.78                  | 700.0       | 10.12                       | 815.0            |
| True    | 655360 x 3072 x 4096         | 26.38                  | 625.0       | 20.61                       | 800.0            |
| False   | 655360 x 2048 x 768          | 3.35                   | 616.0       | 2.93                        | 703.0            |
| False   | 655360 x 1536 x 2048         | 5.65                   | 729.0       | 4.90                        | 842.0            |
| False   | 655360 x 4096 x 1536         | 11.82                  | 698.0       | 11.22                       | 734.0            |
| False   | 655360 x 3072 x 4096         | 24.56                  | 672.0       | 20.68                       | 797.0            |


K-Grouped Gemmn with TMA
| trans_a |   Dimensions (M x N x K,z=128)  | Elapsed Time (ms) | Tflops |
|---------|---------------------------------|-------------------|--------|
| True    | 655360 x 2048 x 768             | 3.44              | 599.0  |
| True    | 655360 x 1536 x 2048            | 5.87              | 703.0  |
| True    | 655360 x 4096 x 1536            | 12.29             | 671.0  |
| True    | 655360 x 3072 x 4096            | 24.99             | 660.0  |


M-Grouped Gemm with mask and TMA(It can't work when trans_b=False)
| trans_b | Matrix Dimensions (N x K x M_masked) | Elapsed Time (ms) | Tflops |
|---------|--------------------------------------|-------------------|--------|
| True    | 2048 x 768 x 101215                  | 0.61              | 526.0  |
| True    | 1536 x 2048 x 101215                 | 0.99              | 642.0  |
| True    | 4096 x 1536 x 101215                 | 2.42              | 525.0  |
| True    | 3072 x 4096 x 101215                 | 4.10              | 622.0  |


## FP8
The following FP8 implementations are available in `gemm/FP8/`:

| Script | Gemm Type | Quantization |
|---------|---------|---------|
|`m_grouped_gemm_channel_expert.py` | M-Grouped Gemm | Per-channel for activation, per-expert for weight  |
|`k_grouped_gemm_channel_expert.py` | K-Grouped Gemm |Per-channel for activation, per-expert for weight  |

The result of FP8 implementations
| Implementation                  | Max Relative Difference | Execution Time (ms) | Tflops |
|---------------------------------|-------------------------|---------------------|--------|
| m_grouped_gemm_channel_expert   | 0.058349609375          | 9.3 ms              | 442.0  |
| k_grouped_gemm_channel_expert   | 0.060546875             | 9.9 ms              | 646.0  |

### Additional Resources

For an alternative FP8 grouped GEMM that quantizes activation with 1x128 tile and weight with 128x128 block, please refer to:
https://github.com/sukoncon/TMA-Adaptive-FP8-Grouped-GEMM.