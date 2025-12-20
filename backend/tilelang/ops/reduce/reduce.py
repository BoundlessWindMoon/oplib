import tilelang
import tilelang.language as T
import torch

@tilelang.jit()
def reduce(n, blocksize, reduce_type="float"):
    
    @T.prim_func
    def reduce_schedule(
        X: T.Tensor(n),
        O: T.Tensor(1),
    ):
        with T.kernel(T.ceildiv(n, blocksize), threads=blocksize) as (bx):
            T.reduce(X, O, reduce_type, dim=0, clear=True)
    
    return reduce_schedule

def reduce_v0(x: torch.Tensor, o: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE_N = 1024
    kernel = reduce(N, BLOCK_SIZE_N)
    kernel(x, o, BLOCK_SIZE_N)
    return o
