import torch

import triton
import triton.language as tl


def generate_configs():
    block_size_n = [16, 32, 64, 128, 256, 512, 1024]
    num_warps = [2, 4, 8, 16]

    configs = []
    for bn in block_size_n:
        for warp in num_warps:
            configs.append(triton.Config({"BLOCK_SIZE_N": bn}, num_warps=warp))
    return configs


@triton.autotune(configs=generate_configs(), key=["N"])
@triton.jit
def vadd_kernel(x_ptr, y_ptr, O_ptr, N, BLOCK_SIZE_N: tl.constexpr):
    tid = tl.program_id(axis=0)
    stride = BLOCK_SIZE_N
    offset = tid * stride + tl.arange(0, BLOCK_SIZE_N)
    mask = offset < N
    a = tl.load(x_ptr + offset, mask=mask, other=0.0)
    b = tl.load(y_ptr + offset, mask=mask, other=0.0)
    c = a + b

    tl.store(O_ptr + offset, c, mask=mask)


def vadd_v0(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(len(x), meta["BLOCK_SIZE_N"]),)
    vadd_kernel[grid](x, y, output, n)
    return output
