import tilelang
import tilelang.language as T
import torch

@tilelang.jit()
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def gemm_schedule(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # T.clear(C_local)
            T.copy(C[by * block_M, bx * block_N], C_local)

            # Auto pipeline the computation
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_schedule

def gemm_v0(A:torch.Tensor, B:torch.Tensor, C:torch.Tensor, M:int, N:int, K:int):
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32    
    
    # -------------------------------
    # Compilation Phase (happens once)
    # -------------------------------
    # TileLang JIT compiler:
    # 1. Analyzes the computation graph
    # 2. Generates optimized CUDA kernels
    # 3. Validates tensor shapes/dtypes
    kernel = matmul(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    # -------------------------------
    # Execution Phase (per call)
    # -------------------------------
    # Launches the compiled kernel with:
    # Grid  : (ceil(N/BLOCK_SIZE_N), ceil(M/BLOCK_SIZE_M))
    # Block : 128 threads
    # Math  : Performs C += A @ B
    kernel(A, B, C)
    return C