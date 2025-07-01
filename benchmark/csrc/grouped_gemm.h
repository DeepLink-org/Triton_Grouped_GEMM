#include <torch/extension.h>

namespace grouped_gemm {

void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b,
		 int available_sm_count=-1);

}  // namespace grouped_gemm
