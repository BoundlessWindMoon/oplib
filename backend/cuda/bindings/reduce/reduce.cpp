#include <torch/extension.h>

void launch_reduce_v0_kernel(torch::Tensor &a, torch::Tensor &sum, int n);

torch::Tensor reduce_v0(torch::Tensor &a, torch::Tensor &sum) {
    TORCH_CHECK(a.is_cuda(), "input tensor 'a' must at CUDA");
    TORCH_CHECK(sum.is_cuda(), "input tensor 'sum' must at CUDA");
    launch_reduce_v0_kernel(a, sum, a.numel());
    return sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_v0", &reduce_v0, "Reduce(CUDA)");
}