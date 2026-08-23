import torch
import triton
import triton.language as tl


@triton.jit
def triton_reduce_sum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    block_sum = tl.sum(x, axis=0)

    tl.atomic_add(out_ptr, block_sum)


def triton_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)

    x = x.contiguous()
    n_elements = x.numel()
    out = torch.zeros((), device=x.device, dtype=torch.float32)

    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    triton_reduce_sum_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    return out
