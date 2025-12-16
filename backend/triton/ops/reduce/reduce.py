import torch
import triton
import triton.language as tl

def generate_configs():
    block_size_n = [32, 64, 128, 256, 512, 1024, 2048]
    num_warps = [4, 8, 16, 32]
    
    configs = []
    for bn in block_size_n:
        for warp in num_warps:
            configs.append(
                triton.Config(
                    {"BLOCK_SIZE_N": bn}, num_warps=warp
                )
            )
    return configs
            
    
@triton.autotune(
    configs=generate_configs(),
    key=["N"],
)

@triton.jit
def reduce_kernel(
    x_ptr,         
    output_ptr,     
    N,              
    BLOCK_SIZE_N: tl.constexpr,  
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    partial_sum = tl.sum(x, axis=0)
    tl.atomic_add(output_ptr, partial_sum)

def reduce_v0(x: torch.Tensor, o: torch.Tensor):        
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    test_o = torch.empty_like(o)
    reduce_kernel[grid](x, test_o, N)
    
    o.zero_()  
    reduce_kernel[grid](x, o, N)  
    return o