import pytest
import torch

from kernels.softmax import triton_softmax
from tests.common import DTYPES, assert_close

DIMS = [1, 13, 255, 256, 257, 1024, 4096]
BATCHES = [1, 4, 33]


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_correctness(device, dtype, batch, dim):
    x = torch.randn(batch, dim, device=device, dtype=dtype)

    out = triton_softmax(x)

    assert out.dtype == x.dtype
    assert out.shape == x.shape
    assert_close(out, torch.softmax(x, dim=-1), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_numerical_stability(device, dtype):
    x = torch.randn(8, 256, device=device, dtype=dtype) * 100

    out = triton_softmax(x)

    assert torch.isfinite(out).all()
    assert_close(out, torch.softmax(x, dim=-1), dtype)


def test_softmax_uniform_row(device):
    x = torch.full((4, 128), 2.5, device=device)

    out = triton_softmax(x)

    assert_close(out, torch.full_like(x, 1.0 / 128), torch.float32)


def test_softmax_rows_sum_to_one(device):
    x = torch.randn(33, 1024, device=device)

    out = triton_softmax(x)

    assert_close(out.sum(dim=-1), torch.ones(33, device=device), torch.float32)


def test_softmax_outputs_non_negative(device):
    x = torch.randn(8, 257, device=device) * 50

    out = triton_softmax(x)

    assert (out >= 0).all()


def test_softmax_non_contiguous_input(device):
    x_base = torch.randn(8, 512, device=device)
    x = x_base[::2]

    assert not x.is_contiguous()
    assert_close(triton_softmax(x), torch.softmax(x, dim=-1), torch.float32)


def test_softmax_does_not_mutate_input(device):
    x = torch.randn(8, 256, device=device)
    x_clone = x.clone()

    out = triton_softmax(x)

    assert torch.equal(x, x_clone)
    assert out.data_ptr() != x.data_ptr()
