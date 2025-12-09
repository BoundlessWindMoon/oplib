#include <torch/extension.h>

void launch_gemm_kernel(torch::Tensor &A, torch::Tensor &B, torch::Tensor &C, int M, int N, int K);

torch::Tensor gemm(torch::Tensor &A, torch::Tensor &B, torch::Tensor &C, int M, int N, int K) {
    TORCH_CHECK(A.is_cuda(), "input tensor A must at CUDA");
    TORCH_CHECK(B.is_cuda(), "input tensor A must at CUDA");
    TORCH_CHECK(C.is_cuda(), "input tensor A must at CUDA");
    
    launch_gemm_kernel(A, B, C, M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "GEMM(CUDA)");
}