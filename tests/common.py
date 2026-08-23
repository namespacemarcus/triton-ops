import torch

TOLERANCES = {
    torch.float32: (1e-6, 1e-6),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}

DTYPES = list(TOLERANCES)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    atol, rtol = TOLERANCES.get(dtype, (1e-6, 1e-6))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
