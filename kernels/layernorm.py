import torch
import triton
import triton.language as tl


@triton.jit
def triton_layer_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_s = tl.program_id(0)
    seq = pid_s * dim
    dims = tl.arange(0, BLOCK_SIZE)
    mask = dims < dim

    x = tl.load(x_ptr + seq + dims, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / dim
    variance = tl.sum(tl.where(mask, (x - mean) * (x - mean), 0.0), axis=0) / dim
    rstd = tl.rsqrt(variance + eps)

    gamma = tl.load(gamma_ptr + dims, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + dims, mask=mask, other=0.0).to(tl.float32)

    out = (x - mean) * rstd * gamma + beta

    tl.store(out_ptr + seq + dims, out, mask=mask)


def triton_layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda
    assert x.dtype == gamma.dtype == beta.dtype

    dim = x.shape[-1]
    assert gamma.numel() == dim
    assert beta.numel() == dim

    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    out = torch.empty_like(x)

    seq_len = x.numel() // dim
    block_size = triton.next_power_of_2(dim)
    triton_layer_norm_kernel[(seq_len,)](
        x,
        gamma,
        beta,
        out,
        dim,
        eps,
        BLOCK_SIZE=block_size,
    )
    return out
