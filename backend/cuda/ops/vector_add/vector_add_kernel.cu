#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void vector_add_kernel(float *c, const float *a, const float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vector_add_kernel(torch::Tensor &c, const torch::Tensor &a, const torch::Tensor &b)
{
    int n = a.numel();
    const int blocksize = 1024;
    const int gridsize = (n + blocksize -1) / blocksize;
    vector_add_kernel<<<gridsize, blocksize>>> (
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
