#include <torch/extension.h>

void launch_vector_add_kernel(torch::Tensor &c, const torch::Tensor &a, const torch::Tensor &b);

torch::Tensor vector_add(const torch::Tensor &a, const torch::Tensor &b)
{
    TORCH_CHECK(a.is_cuda(), "input tensor 'a' must at CUDA ");
    TORCH_CHECK(b.is_cuda(), "input tensor 'b' must at CUDA ");
    TORCH_CHECK(a.numel() == b.numel(), "shape(a) must equal to shape(b)");

    torch::Tensor c = torch::empty_like(a);

    launch_vector_add_kernel(c, a, b);

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add", &vector_add, "Vector Add (CUDA)");
}