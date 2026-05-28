import torch
from torch.library import wrap_triton, triton_op

import triton
import triton.language as tl


@triton.jit
def triton_topk_for_small_k_kernel(
    X_ptr,
    V_ptr,
    I_ptr,
    M,
    stride_xn,
    stride_xm,
    stride_vn,
    stride_vk,
    stride_in,
    stride_ik,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LARGEST: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_M)
    mask = offsets < M

    x_ptrs = X_ptr + pid * stride_xn + offsets * stride_xm

    neg_inf = float("-inf")
    pos_inf = float("inf")

    if LARGEST:
        x = tl.load(x_ptrs, mask=mask, other=neg_inf)
    else:
        x = tl.load(x_ptrs, mask=mask, other=pos_inf)

    for k_idx in tl.static_range(K):
        if LARGEST:
            idx = tl.argmax(x, axis=0)
            val = tl.max(x, axis=0)
        else:
            idx = tl.argmin(x, axis=0)
            val = tl.min(x, axis=0)

        tl.store(V_ptr + pid * stride_vn + k_idx * stride_vk, val)
        tl.store(I_ptr + pid * stride_in + k_idx * stride_ik, idx.to(tl.int64))

        if LARGEST:
            x = tl.where(offsets == idx, neg_inf, x)
        else:
            x = tl.where(offsets == idx, pos_inf, x)


@triton_op("customop::triton_topk_for_small_k", mutates_args={})
def triton_topk_for_small_k(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    ndim = input.dim()
    if dim < 0:
        dim += ndim

    moved = dim != ndim - 1
    if moved:
        perm = list(range(ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = input.permute(perm).contiguous()
    else:
        x = input.contiguous()

    leading_shape = x.shape[:-1]
    m = x.shape[-1]
    n = 1
    for size in leading_shape:
        n *= size

    x2d = x.view(n, m)
    values = torch.empty((n, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((n, k), dtype=torch.int64, device=x.device)

    block_m = max(triton.next_power_of_2(m), 16)
    grid = (n,)
    wrap_triton(triton_topk_for_small_k_kernel)[grid](
        X_ptr=x2d,
        V_ptr=values,
        I_ptr=indices,
        M=m,
        stride_xn=x2d.stride(0),
        stride_xm=x2d.stride(1),
        stride_vn=values.stride(0),
        stride_vk=values.stride(1),
        stride_in=indices.stride(0),
        stride_ik=indices.stride(1),
        K=k,
        BLOCK_M=block_m,
        LARGEST=largest,
    )

    out_shape = list(leading_shape) + [k]
    values = values.view(out_shape)
    indices = indices.view(out_shape)

    if moved:
        perm_back = list(range(ndim))
        perm_back[dim], perm_back[-1] = perm_back[-1], perm_back[dim]
        values = values.permute(perm_back).contiguous()
        indices = indices.permute(perm_back).contiguous()

    _ = sorted

    return values, indices
