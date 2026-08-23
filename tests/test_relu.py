import pytest
import torch

from kernels.relu import triton_relu
from tests.common import DTYPES, assert_close

SIZES = [1, 255, 256, 257, 1024, 4096, 1_000_003]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_relu_correctness(device, size, dtype):
    x = torch.randn(size, device=device, dtype=dtype) * 10

    out = triton_relu(x)

    assert_close(out, torch.relu(x), dtype)


def test_relu_all_negative(device):
    x = -torch.rand(1024, device=device) - 1e-3

    out = triton_relu(x)

    assert_close(out, torch.zeros_like(x), torch.float32)


def test_relu_all_positive(device):
    x = torch.rand(1024, device=device) + 1e-3

    out = triton_relu(x)

    assert_close(out, x, torch.float32)


def test_relu_zeros(device):
    x = torch.zeros(512, device=device)

    assert_close(triton_relu(x), x, torch.float32)


def test_relu_multidim(device):
    x = torch.randn(4, 8, 16, device=device)

    assert_close(triton_relu(x), torch.relu(x), torch.float32)


def test_relu_non_contiguous_input(device):
    x_base = torch.randn(256, device=device)
    x = x_base[::2]

    assert not x.is_contiguous()
    assert_close(triton_relu(x), torch.relu(x), torch.float32)


def test_relu_does_not_mutate_input(device):
    x = torch.randn(512, device=device)
    x_clone = x.clone()

    out = triton_relu(x)

    assert_close(x, x_clone, torch.float32)
    assert out.data_ptr() != x.data_ptr()
