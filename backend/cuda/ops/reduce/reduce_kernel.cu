#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdio.h>
__global__ void reduce_v0_kernel(float* input, float* sum, int n) {
    extern __shared__ float smem[];

    // global -> smem
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    smem[tid] = (idx < n ? input[idx] : 0.0f);

    __syncthreads();
    for(int s = 1; s < blockDim.x; s <<= 1) {
        int start = tid * 2 * s;
        if (start < blockDim.x) {
            smem[start] = smem[start] + smem[start + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        // printf("blockidx ==%d sum = %.8f\n", blockIdx.x, smem[0]); 
        atomicAdd(sum, smem[0]);
    }

    __syncthreads();
}

void launch_reduce_v0_kernel(torch::Tensor &input, torch::Tensor &sum, int n) {
    const int blocksize = 32;
    dim3 block(blocksize);
    dim3 grid((n + blocksize-1) / blocksize);

    reduce_v0_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        sum.data_ptr<float>(),
        n
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}