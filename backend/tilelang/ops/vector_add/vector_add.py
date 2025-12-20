import tilelang
import tilelang.language as T
import torch

@tilelang.jit()
def vadd(n, blocksize, dtype="float", accum_dtype="float"):
    
    @T.prim_func
    def vadd_schedule(
        A: T.Tensor(n),
        B: T.Tensor(n),
        O: T.Tensor(n),
    ):
        with T.kernel(T.ceildiv(n, blocksize), threads=blocksize) as (bx):
            A_shared = T.alloc_shared(blocksize, dtype)
            B_shared = T.alloc_shared(blocksize, dtype)
            O_local = T.alloc_fragment(blocksize, accum_dtype)
            
            T.copy(A[bx * blocksize], A_shared)
            T.copy(B[bx * blocksize], B_shared)
            for i in T.serial(blocksize):
                O_local[i] = A_shared[i] + B_shared[i]
                
            T.copy(O_local, O[bx * blocksize])
    return vadd_schedule    

def vadd_v0(x: torch.Tensor, y: torch.Tensor):
    BLOCK_SIZE_N = 512
    output = torch.empty_like(x)
    n = x.numel()
    
    kernel = vadd(n, BLOCK_SIZE_N)
    kernel(x, y, output)
    return output  
    