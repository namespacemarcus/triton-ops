import torch
import triton
import triton.language as tl


@triton.jit
def triton_softmax_kernel(
    x_ptr,
    out_ptr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_s = tl.program_id(0)
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    mask = dim_offsets < dim
    seq_start = pid_s * dim

    x = tl.load(
        x_ptr + seq_start + dim_offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    max_value = tl.max(x, axis=0)
    out = tl.exp(x - max_value) / tl.sum(tl.exp(x - max_value), axis=0)

    tl.store(out_ptr + seq_start + dim_offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.shape[-1] > 0

    seq_len, dim = x.shape
    assert seq_len > 0 and dim > 0

    x = x.contiguous()
    out = torch.empty_like(x)

    block_size = triton.next_power_of_2(dim)
    triton_softmax_kernel[(seq_len,)](
        x,
        out,
        dim=dim,
        BLOCK_SIZE=block_size,
    )
    return out
