
import pytest
import torch
from kernels.topk_for_small_k.topk_for_small_k import topk_for_small_k


class TestTopkForSmallK:
    def test_basic(self):
        x = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]], device="cuda")
        k = 3
        values, indices = topk_for_small_k(x, k=k, dim=-1, largest=True)

        expected_values = torch.tensor([[5.0, 4.0, 3.0]], device="cuda")
        expected_indices = torch.tensor([[1, 3, 4]], device="cuda", dtype=torch.int64)

        assert torch.equal(values, expected_values)
        assert torch.equal(indices, expected_indices)

    @pytest.mark.parametrize(
        "shape,k",
        [
            ((10,), 5),
            ((100,), 10),
            ((1, 50), 10),
            ((4, 32), 8),
            ((2, 4, 64), 16),
        ],
    )
    def test_shapes(self, shape, k):
        torch.manual_seed(42)
        x = torch.randn(*shape, device="cuda")

        values, indices = topk_for_small_k(x, k=k, dim=-1, largest=True)

        assert values.shape == indices.shape
        assert values.shape[-1] == k

    def test_dim_0(self):
        x = torch.randn(8, 16, device="cuda")
        k = 4
        values, indices = topk_for_small_k(x, k=k, dim=0, largest=True)

        assert values.shape == (4, 16)
        assert indices.shape == (4, 16)

    def test_dim_1(self):
        x = torch.randn(8, 16, device="cuda")
        k = 4
        values, indices = topk_for_small_k(x, k=k, dim=1, largest=True)

        assert values.shape == (8, 4)
        assert indices.shape == (8, 4)

    def test_largest_true(self):
        x = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]], device="cuda")
        values, indices = topk_for_small_k(x, k=3, dim=-1, largest=True)

        assert values[0, 0] == 5.0
        assert values[0, 1] == 4.0
        assert values[0, 2] == 3.0

    def test_largest_false(self):
        x = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]], device="cuda")
        values, indices = topk_for_small_k(x, k=3, dim=-1, largest=False)

        assert values[0, 0] == 1.0
        assert values[0, 1] == 2.0
        assert values[0, 2] == 3.0

    def test_k_equals_m(self):
        x = torch.randn(4, 16, device="cuda")
        k = 16
        values, indices = topk_for_small_k(x, k=k, dim=-1, largest=True)

        assert values.shape == (4, 16)

    def test_k_equals_1(self):
        x = torch.randn(4, 16, device="cuda")
        values, indices = topk_for_small_k(x, k=1, dim=-1, largest=True)

        assert values.shape == (4, 1)
        assert indices.shape == (4, 1)

    def test_compare_with_torch(self):
        torch.manual_seed(123)
        x = torch.randn(16, 32, 64, device="cuda")
        k = 8

        triton_values, triton_indices = topk_for_small_k(x, k=k, dim=-1, largest=True)
        torch_values, torch_indices = torch.topk(x, k=k, dim=-1, largest=True)

        assert torch.allclose(triton_values, torch_values, atol=1e-3)
        assert torch.equal(triton_indices, torch_indices)

    def test_compare_largest_false(self):
        torch.manual_seed(123)
        x = torch.randn(8, 32, device="cuda")
        k = 4

        triton_values, triton_indices = topk_for_small_k(x, k=k, dim=-1, largest=False)
        torch_values, torch_indices = torch.topk(x, k=k, dim=-1, largest=False)

        assert torch.allclose(triton_values, torch_values, atol=1e-3)
        assert torch.equal(triton_indices, torch_indices)

    def test_different_last_dim(self):
        k = 4
        for m in [16, 32, 64, 128]:
            x = torch.randn(4, m, device="cuda")
            values, indices = topk_for_small_k(x, k=k, dim=-1, largest=True)

            assert values.shape[-1] == k
            assert indices.shape[-1] == k

    def test_negative_dim(self):
        x = torch.randn(4, 16, device="cuda")
        k = 4

        values1, indices1 = topk_for_small_k(x, k=k, dim=-1, largest=True)
        values2, indices2 = topk_for_small_k(x, k=k, dim=1, largest=True)

        assert torch.equal(values1, values2)
        assert torch.equal(indices1, indices2)


class TestTopkForSmallKErrors:
    def test_not_cuda(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(AssertionError, match="input must be a CUDA tensor"):
            topk_for_small_k(x, k=2)

    def test_k_zero(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        with pytest.raises(AssertionError, match="k must be greater than 0"):
            topk_for_small_k(x, k=0)

    def test_k_negative(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        with pytest.raises(AssertionError, match="k must be greater than 0"):
            topk_for_small_k(x, k=-1)

    def test_k_too_large(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        with pytest.raises(AssertionError, match="k .* must be <= size of dim"):
            topk_for_small_k(x, k=5)

    def test_dim_out_of_range(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        with pytest.raises(AssertionError, match="dim out of range"):
            topk_for_small_k(x, k=2, dim=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
