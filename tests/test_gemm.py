import pytest
import torch

from kernels.gemm import triton_gemm

SHAPES = [
    (32, 32, 32),
    (64, 64, 64),
    (1, 1, 1),
    (17, 13, 37),
    (130, 65, 100),
    (256, 128, 512),
]

DTYPES = [torch.float16, torch.bfloat16]

RTOL = {torch.float16: 1e-2, torch.bfloat16: 2e-2}


def matmul_ref(a, b):
    return (a.float() @ b.float()).to(a.dtype)


def assert_gemm_close(actual, a, b):
    expected = matmul_ref(a, b)
    torch.testing.assert_close(actual, expected, rtol=RTOL[a.dtype], atol=1e-2)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm_correctness(device, dtype, shape):
    m, n, k = shape
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)

    out = triton_gemm(a, b)

    assert out.dtype == a.dtype
    assert out.shape == (m, n)
    assert_gemm_close(out, a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm_transposed_b(device, dtype):
    a = torch.randn(64, 96, device=device, dtype=dtype)
    b = torch.randn(64, 96, device=device, dtype=dtype).t()

    assert not b.is_contiguous()
    assert_gemm_close(triton_gemm(a, b), a, b)


def test_gemm_strided_a(device):
    a_base = torch.randn(128, 64, device=device, dtype=torch.float16)
    a = a_base[::2]
    b = torch.randn(64, 64, device=device, dtype=torch.float16)

    assert not a.is_contiguous()
    assert_gemm_close(triton_gemm(a, b), a, b)


def test_gemm_does_not_mutate_inputs(device):
    a = torch.randn(64, 64, device=device, dtype=torch.float16)
    b = torch.randn(64, 64, device=device, dtype=torch.float16)
    a_clone, b_clone = a.clone(), b.clone()

    out = triton_gemm(a, b)

    assert torch.equal(a, a_clone)
    assert torch.equal(b, b_clone)
    assert out.data_ptr() != a.data_ptr()


def test_gemm_rejects_shape_mismatch(device):
    a = torch.randn(64, 32, device=device, dtype=torch.float16)
    b = torch.randn(64, 64, device=device, dtype=torch.float16)

    with pytest.raises(AssertionError):
        triton_gemm(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_gemm_rejects_unsupported_dtypes(device, dtype):
    a = torch.zeros(64, 32, device=device, dtype=dtype)
    b = torch.zeros(32, 64, device=device, dtype=dtype)

    with pytest.raises(AssertionError):
        triton_gemm(a, b)


def test_gemm_rejects_mixed_dtypes(device):
    a = torch.randn(64, 32, device=device, dtype=torch.float16)
    b = torch.randn(32, 64, device=device, dtype=torch.bfloat16)

    with pytest.raises(AssertionError):
        triton_gemm(a, b)


@pytest.mark.parametrize("ndim", [1, 3])
def test_gemm_rejects_non_2d(device, ndim):
    a = torch.randn((64,) * ndim, device=device, dtype=torch.float16)
    b = torch.randn(64, 64, device=device, dtype=torch.float16)

    with pytest.raises(AssertionError):
        triton_gemm(a, b)
