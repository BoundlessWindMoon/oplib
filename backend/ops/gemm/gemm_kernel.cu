#include <cuda_runtime.h>
#include <torch/extension.h>

// __global__ void gemm_kernel(half *A, half *B, half *C, int M, int N, int K) {
//     const int BM = 64;
//     const int BN = 64;
//     const int BK = 64;
//     const int TM = 8;
//     const int TN = 8;
//     const int TK = 2;

//     __shared__ half smem_a[BM][BK];
//     __shared__ half smem_b[BK][BN];

//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     int tid = ty * blockIdx.x + tx;
//     int warp_idx = tid / 32;

//     // gmem -> smem
//     int load_a_smem_m = ;
//     int load_smem_k = ;
//     int load_b_smem_n = ;

//     int load_a_gmem_m = ;
//     int load_gmem_k = ;
//     int load_b_gmem_n = ;

//     smem_a[load_a_smem_m][load_smem_k] = ;
//     smem_b[load_smem_k][load_b_smem_n] = ;

//     __syncthreads();

//     for(int bk = 0; bk < (K + BK-1) / BK; bk++) {

//     }
//     __syncthreads();
// }

__global__ void gemm_kernel(__half *A, __half *B, __half *C, int M, int N, int K) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M || c >= N) return;
    float value = 0.0f;  
    for (int k = 0; k < K; ++k) {
        value += __half2float(A[r * K + k]) * __half2float(B[k * N + c]);
    }
    C[r * N + c] = __hadd(C[r * N + c], __float2half(value));  
}

void launch_gemm_kernel(torch::Tensor &A, torch::Tensor &B, torch::Tensor &C, int M, int N, int K)
{
    int BM = 16;
    int BN = 16;

    int TM = 1;
    int TN = 1;

    dim3 block(16, 16);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    gemm_kernel<<<grid, block>>>(
        reinterpret_cast<__half*>(A.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(B.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<torch::Half>()),
        M,
        N,
        K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}