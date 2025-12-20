#include <torch/extension.h>

torch::Tensor launch_attn_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

torch::Tensor attn_v0(torch::Tensor &Q, torch::Tensor &K, torch::Tensor &V) {
    TORCH_CHECK(Q.is_cuda(), "input tensor A must at CUDA");
    TORCH_CHECK(K.is_cuda(), "input tensor A must at CUDA");
    TORCH_CHECK(V.is_cuda(), "input tensor A must at CUDA");
    
    return launch_attn_kernel(Q, K, V);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_v0", &attn_v0, "Attention(CUDA)");
}