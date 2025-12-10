#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void vector_add_v0_kernel(float *c, const float *a, const float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// use smem and register
__global__ void vector_add_v1_kernel(float *c, const float *a, const float *b, int n)
{
    __shared__ float smem_a[1024 * 4];
    __shared__ float smem_b[1024 * 4];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float c_tile[4] = {0.0f};

    // gmem -> smem
    if(idx + 3 < n) {
        smem_a[tid * 4 + 0] = a[idx * 4 + 0];
        smem_a[tid * 4 + 1] = a[idx * 4 + 1];
        smem_a[tid * 4 + 2] = a[idx * 4 + 2];
        smem_a[tid * 4 + 3] = a[idx * 4 + 3];
        smem_b[tid * 4 + 0] = b[idx * 4 + 0];
        smem_b[tid * 4 + 1] = b[idx * 4 + 1];
        smem_b[tid * 4 + 2] = b[idx * 4 + 2];
        smem_b[tid * 4 + 3] = b[idx * 4 + 3];
    } else {
        smem_a[tid * 4 + 0] = (idx + 0 < n ? a[idx * 4 + 0] : 0.0f); 
        smem_a[tid * 4 + 1] = (idx + 1 < n ? a[idx * 4 + 1] : 0.0f); 
        smem_a[tid * 4 + 2] = (idx + 2 < n ? a[idx * 4 + 2] : 0.0f); 
        smem_a[tid * 4 + 3] = (idx + 3 < n ? a[idx * 4 + 3] : 0.0f); 

        smem_b[tid * 4 + 0] = (idx + 0 < n ? b[idx * 4 + 0] : 0.0f); 
        smem_b[tid * 4 + 1] = (idx + 1 < n ? b[idx * 4 + 1] : 0.0f); 
        smem_b[tid * 4 + 2] = (idx + 2 < n ? b[idx * 4 + 2] : 0.0f); 
        smem_b[tid * 4 + 3] = (idx + 3 < n ? b[idx * 4 + 3] : 0.0f); 
    }

    __syncthreads();

    if (idx + 3 < n) {
        c_tile[tid * 4 + 0] = smem_a[tid * 4 + 0] + smem_b[tid * 4 + 0];
        c_tile[tid * 4 + 1] = smem_a[tid * 4 + 1] + smem_b[tid * 4 + 1];
        c_tile[tid * 4 + 2] = smem_a[tid * 4 + 2] + smem_b[tid * 4 + 2];
        c_tile[tid * 4 + 3] = smem_a[tid * 4 + 3] + smem_b[tid * 4 + 3];
    } else {
        c_tile[tid * 4 + 0] = (idx + 0 < n ? smem_a[tid * 4 + 0] + smem_b[tid * 4 + 0] : 0.0f);
        c_tile[tid * 4 + 1] = (idx + 1 < n ? smem_a[tid * 4 + 1] + smem_b[tid * 4 + 1] : 0.0f);
        c_tile[tid * 4 + 2] = (idx + 2 < n ? smem_a[tid * 4 + 2] + smem_b[tid * 4 + 2] : 0.0f);
        c_tile[tid * 4 + 3] = (idx + 3 < n ? smem_a[tid * 4 + 3] + smem_b[tid * 4 + 3] : 0.0f);
    }
    __syncthreads();

    // smem -> gmem
    c[idx * 4 + 0] = c_tile[tid * 4 + 0];
    c[idx * 4 + 1] = c_tile[tid * 4 + 1];
    c[idx * 4 + 2] = c_tile[tid * 4 + 2];
    c[idx * 4 + 3] = c_tile[tid * 4 + 3];
    __syncthreads();
}

void launch_vector_add_kernel(torch::Tensor &c, const torch::Tensor &a, const torch::Tensor &b)
{
    int n = a.numel();
    // const int blocksize = 1024;
    // const int gridsize = (n + blocksize -1) / blocksize ;

    // version 0：naive implementation
    // vector_add_v0_kernel<<<gridsize, blocksize>>> (
    //     c.data_ptr<float>(),
    //     a.data_ptr<float>(),
    //     b.data_ptr<float>(),
    //     n
    // );
    

    const int blocksize = 1024;
    const int gridsize = (n + blocksize -1) / blocksize / 4;

    vector_add_v1_kernel<<<gridsize, blocksize>>> (
        c.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        n
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
