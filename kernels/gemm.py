import torch
import triton
import triton.language as tl

_GEMM_CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": 32,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
        },
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "BLOCK_K": 32,
        },
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
        },
        num_warps=8,
        num_stages=4,
    ),
]


@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def triton_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k_block * BLOCK_K + offs_k
        k_mask = k_offsets < K

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )

        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype
    assert a.dtype in (torch.float16, torch.bfloat16)

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    triton_gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c
