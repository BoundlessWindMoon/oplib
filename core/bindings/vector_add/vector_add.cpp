#include <torch/extension.h>

void launch_vector_add_kernel(torch::Tensor &c, const torch::Tensor &a, const torch::Tensor &b);

torch::Tensor vector_add(const torch::Tensor &a, const torch::Tensor &b)
{
    TORCH_CHECK(a.is_cuda(), "输入张量 'a' 必须在 CUDA 上");
    TORCH_CHECK(b.is_cuda(), "输入张量 'b' 必须在 CUDA 上");
    TORCH_CHECK(a.numel() == b.numel(), "输入张量必须有相同的元素数量");

    torch::Tensor c = torch::empty_like(a);

    launch_vector_add_kernel(c, a, b);

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add", &vector_add, "Vector Add (CUDA)");
}