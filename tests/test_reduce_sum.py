import pytest
import torch

from kernels.reduce_sum import triton_reduce_sum
from tests.common import DTYPES

SIZES = [1, 255, 256, 257, 1024, 4096, 100_003]

RTOL = {
    torch.float32: 1e-4,
    torch.float16: 1e-3,
    torch.bfloat16: 1e-2,
}


def assert_sum_close(out, x):
    expected = x.double().sum().to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=RTOL[x.dtype], atol=1e-3)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_sum_correctness(device, size, dtype):
    x = torch.randn(size, device=device, dtype=dtype)

    out = triton_reduce_sum(x)

    assert out.dtype == torch.float32
    assert out.shape == ()
    assert_sum_close(out, x)


def test_reduce_sum_zeros(device):
    x = torch.zeros(4096, device=device)

    assert_sum_close(triton_reduce_sum(x), x)


def test_reduce_sum_ones(device):
    x = torch.ones(4096, device=device)

    assert_sum_close(triton_reduce_sum(x), x)


def test_reduce_sum_negative(device):
    x = -torch.rand(4096, device=device) - 1e-3

    assert_sum_close(triton_reduce_sum(x), x)


def test_reduce_sum_multidim(device):
    x = torch.randn(4, 8, 16, device=device)

    assert_sum_close(triton_reduce_sum(x), x)


def test_reduce_sum_non_contiguous_input(device):
    x_base = torch.randn(256, device=device)
    x = x_base[::2]

    assert not x.is_contiguous()
    assert_sum_close(triton_reduce_sum(x), x)


def test_reduce_sum_does_not_mutate_input(device):
    x = torch.randn(512, device=device)
    x_clone = x.clone()

    out = triton_reduce_sum(x)

    assert torch.equal(x, x_clone)
    assert out.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_reduce_sum_rejects_int_dtypes(device, dtype):
    x = torch.zeros(64, device=device, dtype=dtype)

    with pytest.raises(AssertionError):
        triton_reduce_sum(x)
