import pytest
import torch

from kernels.layernorm import triton_layer_norm
from tests.common import DTYPES, assert_close

DIMS = [1, 13, 255, 256, 257, 1024, 4096]
BATCH_SHAPES = [(), (4,), (2, 3)]


def ref_layer_norm(x, gamma, beta, eps):
    return torch.nn.functional.layer_norm(
        x, x.shape[-1:], weight=gamma, bias=beta, eps=eps
    )


def make_inputs(device, dtype, shape, dim):
    x = torch.randn(*shape, dim, device=device, dtype=dtype)
    gamma = torch.randn(dim, device=device, dtype=dtype)
    beta = torch.randn(dim, device=device, dtype=dtype)
    return x, gamma, beta


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("shape", BATCH_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_layernorm_correctness(device, dtype, shape, dim):
    x, gamma, beta = make_inputs(device, dtype, shape, dim)

    out = triton_layer_norm(x, gamma, beta)

    assert out.dtype == x.dtype
    assert out.shape == x.shape
    assert_close(out, ref_layer_norm(x, gamma, beta, 1e-5), dtype)


@pytest.mark.parametrize("eps", [1e-5, 1e-3, 1e-1])
def test_layernorm_eps(device, eps):
    x, gamma, beta = make_inputs(device, torch.float32, (8,), 256)

    assert_close(
        triton_layer_norm(x, gamma, beta, eps=eps),
        ref_layer_norm(x, gamma, beta, eps),
        torch.float32,
    )


def test_layernorm_constant_row(device):
    x = torch.full((4, 128), 3.14, device=device)
    gamma = torch.ones(128, device=device)
    beta = torch.zeros(128, device=device)

    out = triton_layer_norm(x, gamma, beta, eps=1e-5)

    assert_close(out, ref_layer_norm(x, gamma, beta, 1e-5), torch.float32)


def test_layernorm_zero_variance_stability(device):
    x = torch.full((4, 128), 2.0, device=device)
    gamma = torch.ones(128, device=device)
    beta = torch.zeros(128, device=device)

    out = triton_layer_norm(x, gamma, beta)

    assert torch.isfinite(out).all()


def test_layernorm_non_contiguous_input(device):
    x, gamma, beta = make_inputs(device, torch.float32, (4, 64), 128)
    x = x[::2]

    assert not x.is_contiguous()
    assert_close(
        triton_layer_norm(x, gamma, beta),
        ref_layer_norm(x, gamma, beta, 1e-5),
        torch.float32,
    )


def test_layernorm_does_not_mutate_inputs(device):
    x, gamma, beta = make_inputs(device, torch.float32, (4,), 256)
    clones = [t.clone() for t in (x, gamma, beta)]

    out = triton_layer_norm(x, gamma, beta)

    assert torch.equal(x, clones[0])
    assert torch.equal(gamma, clones[1])
    assert torch.equal(beta, clones[2])
    assert out.data_ptr() != x.data_ptr()


def test_layernorm_rejects_bad_weight_size(device):
    x, gamma, beta = make_inputs(device, torch.float32, (4,), 128)

    with pytest.raises(AssertionError):
        triton_layer_norm(x, gamma[:127], beta)
    with pytest.raises(AssertionError):
        triton_layer_norm(x, gamma, beta[:127])


@pytest.mark.parametrize(
    "dtype_pair", [(torch.float32, torch.float16), (torch.float16, torch.float32)]
)
def test_layernorm_rejects_mixed_dtypes(device, dtype_pair):
    x = torch.randn(4, 128, device=device, dtype=dtype_pair[0])
    gamma = torch.randn(128, device=device, dtype=dtype_pair[1])
    beta = torch.randn(128, device=device, dtype=dtype_pair[0])

    with pytest.raises(AssertionError):
        triton_layer_norm(x, gamma, beta)
