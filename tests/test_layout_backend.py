import unittest
import sys
from pathlib import Path

import torch

_REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "gemm").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gemm.backends import GemmBackend, normalize_backend, select_backend
from gemm.layout import (
    make_deep_gemm_grouped_layout,
    make_deep_gemm_psum_layout,
    make_group_starts,
    make_grouped_layout,
    round_up_to_alignment,
    validate_total_m,
)


class LayoutBackendTest(unittest.TestCase):
    def test_round_up_to_alignment(self):
        sizes = torch.tensor([0, 1, 127, 128, 129], dtype=torch.int32)
        self.assertEqual(round_up_to_alignment(sizes, 128).tolist(), [0, 128, 128, 128, 256])

    def test_make_group_starts_logical_and_padded(self):
        sizes = torch.tensor([3, 5, 0, 7], dtype=torch.int32)
        self.assertEqual(make_group_starts(sizes, padded=False, block_m=4).tolist(), [0, 3, 8, 8])
        self.assertEqual(make_group_starts(sizes, padded=True, block_m=4).tolist(), [0, 4, 12, 12])

    def test_make_grouped_layout_summary(self):
        layout = make_grouped_layout([3, 5, 0, 7], block_m=4)
        self.assertEqual(layout.sizes.tolist(), [3, 5, 0, 7])
        self.assertEqual(layout.padded_sizes.tolist(), [4, 8, 0, 8])
        self.assertEqual(layout.starts.tolist(), [0, 4, 12, 12])
        self.assertEqual(int(layout.total_m.item()), 15)
        self.assertEqual(int(layout.total_padded_m.item()), 20)

    def test_make_deep_gemm_grouped_layout(self):
        self.assertEqual(
            make_deep_gemm_grouped_layout([3, 5], block_m=4).tolist(),
            [0, 0, 0, -1, 1, 1, 1, 1, 1, -1, -1, -1],
        )

    def test_make_deep_gemm_psum_layout(self):
        self.assertEqual(make_deep_gemm_psum_layout([3, 5], block_m=4).tolist(), [3, 9])

    def test_validate_total_m(self):
        validate_total_m([2, 3, 4], 9)
        with self.assertRaisesRegex(ValueError, r"sum\(size_per_group\)"):
            validate_total_m([2, 3, 4], 10)

    def test_backend_normalization_and_triton_selection(self):
        self.assertEqual(normalize_backend("triton"), GemmBackend.TRITON)
        self.assertEqual(normalize_backend("deepgemm"), GemmBackend.DEEP_GEMM)
        self.assertEqual(select_backend("triton"), GemmBackend.TRITON)
        with self.assertRaisesRegex(ValueError, "Unsupported GEMM backend"):
            normalize_backend("cutlass")


if __name__ == "__main__":
    unittest.main()
