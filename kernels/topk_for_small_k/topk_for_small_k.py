import torch

from .triton_topk_for_small_k import triton_topk_for_small_k


def topk_for_small_k(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(input, torch.Tensor), "input must be a torch.Tensor."
    assert input.is_cuda, "input must be a CUDA tensor."
    assert k > 0, "k must be greater than 0."

    ndim = input.dim()
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim, "dim out of range."

    m = input.shape[dim]
    assert k <= m, f"k ({k}) must be <= size of dim({m})."

    return triton_topk_for_small_k(input, k, dim, largest, sorted)
