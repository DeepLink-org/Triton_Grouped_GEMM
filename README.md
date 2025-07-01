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

K-Grouped Gemm
| trans_b | trans_a | Matrix Dimensions (m x n x K) | Elapsed Time (ms) | Tflops |
|---------|---------|-------------------------------|-------------------|--------|
| True    | True    | 1536 x 2048 x 524288          | 4.75              | 694.0  |
| True    | True    | 2048 x 768 x 524288           | 2.78              | 594.0  |
| True    | True    | 3072 x 4096 x 524288          | 19.58             | 674.0  |
| True    | True    | 4096 x 1536 x 524288          | 10.36             | 637.0  |
| False   | True    | 1536 x 2048 x 524288          | 4.7               | 701.0  |
| False   | True    | 2048 x 768 x 524288           | 2.53              | 652.0  |
| False   | True    | 3072 x 4096 x 524288          | 19.73             | 669.0  |
| False   | True    | 4096 x 1536 x 524288          | 10.85             | 608.0  |


M-Grouped Gemm
| trans_b | Matrix Dimensions (n x k x M) | Elapsed Time (ms) | Tflops |
|---------|-------------------------------|-------------------|--------|
| True    | 1536 x 2048 x 524288          | 4.61              | 716.0  |
| True    | 2048 x 768 x 524288           | 2.58              | 639.0  |
| True    | 3072 x 4096 x 524288          | 17.67             | 747.0  |
| True    | 4096 x 1536 x 524288          | 9.27              | 712.0  |
| False   | 1536 x 2048 x 524288          | 4.51              | 731.0  |
| False   | 2048 x 768 x 524288           | 2.69              | 612.0  |
| False   | 3072 x 4096 x 524288          | 18.48             | 714.0  |
| False   | 4096 x 1536 x 524288          | 9.53              | 693.0  |



M-Grouped Gemm with mask(It can't work when trans_b=False)
| trans_b | Matrix Dimensions (n x k x M_masked) | Elapsed Time (ms) | Tflops |
|---------|--------------------------------------|-------------------|--------|
| True    | 1536 x 2048 x 98325                  | 1.02              | 608.0  |
| True    | 2048 x 768 x 98325                   | 0.58              | 533.0  |
| True    | 3072 x 4096 x 98325                  | 4.02              | 616.0  |
| True    | 4096 x 1536 x 98325                  | 2.27              | 544.0  |


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