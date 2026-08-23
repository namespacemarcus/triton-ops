import torch
import triton
import triton.language as tl


@triton.jit
def triton_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(x, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda

    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    triton_relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=256)
    return out
