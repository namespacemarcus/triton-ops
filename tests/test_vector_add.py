import pytest
import torch

from kernels.vector_add import triton_vector_add
from tests.common import DTYPES, assert_close

SIZES = [1, 255, 256, 257, 1024, 4096, 1_000_003]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_add_correctness(device, size, dtype):
    x = torch.randn(size, device=device, dtype=dtype)
    y = torch.randn(size, device=device, dtype=dtype)

    out = triton_vector_add(x, y)

    assert_close(out, x + y, dtype)


def test_vector_add_multidim(device):
    x = torch.randn(8, 32, device=device)
    y = torch.randn(8, 32, device=device)

    assert_close(triton_vector_add(x, y), x + y, torch.float32)


def test_vector_add_non_contiguous_inputs(device):
    x_base = torch.randn(128, device=device)
    y_base = torch.randn(128, device=device)
    x = x_base[::2]
    y = y_base.view(64, 2)[:, 0]

    assert not x.is_contiguous() and not y.is_contiguous()
    assert_close(triton_vector_add(x, y), x + y, torch.float32)


def test_vector_add_does_not_mutate_inputs(device):
    x = torch.randn(512, device=device)
    y = torch.randn(512, device=device)
    x_clone, y_clone = x.clone(), y.clone()

    out = triton_vector_add(x, y)

    assert_close(x, x_clone, torch.float32)
    assert_close(y, y_clone, torch.float32)
    assert out.data_ptr() != x.data_ptr()


def test_vector_add_rejects_shape_mismatch(device):
    x = torch.randn(16, device=device)
    y = torch.randn(8, device=device)

    with pytest.raises(AssertionError):
        triton_vector_add(x, y)


@pytest.mark.parametrize(
    "dtype_pair", [(torch.float32, torch.float16), (torch.float16, torch.float32)]
)
def test_vector_add_rejects_mixed_dtypes(device, dtype_pair):
    x = torch.randn(16, device=device, dtype=dtype_pair[0])
    y = torch.randn(16, device=device, dtype=dtype_pair[1])

    with pytest.raises(AssertionError):
        triton_vector_add(x, y)
