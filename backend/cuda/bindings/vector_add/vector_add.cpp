#include <torch/extension.h>

void launch_vector_add_v0_kernel(torch::Tensor &c, const torch::Tensor &a, const torch::Tensor &b);
void launch_vector_add_v1_kernel(torch::Tensor &c, const torch::Tensor &a, const torch::Tensor &b);

torch::Tensor vadd_v0(const torch::Tensor &a, const torch::Tensor &b)
{
    TORCH_CHECK(a.is_cuda(), "input tensor 'a' must at CUDA ");
    TORCH_CHECK(b.is_cuda(), "input tensor 'b' must at CUDA ");
    TORCH_CHECK(a.numel() == b.numel(), "shape(a) must equal to shape(b)");

    torch::Tensor c = torch::empty_like(a);

    launch_vector_add_v0_kernel(c, a, b);

    return c;
}

torch::Tensor vadd_v1(const torch::Tensor &a, const torch::Tensor &b)
{
    TORCH_CHECK(a.is_cuda(), "input tensor 'a' must at CUDA ");
    TORCH_CHECK(b.is_cuda(), "input tensor 'b' must at CUDA ");
    TORCH_CHECK(a.numel() == b.numel(), "shape(a) must equal to shape(b)");

    torch::Tensor c = torch::empty_like(a);

    launch_vector_add_v1_kernel(c, a, b);

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vadd_v0", &vadd_v0, "Vector Add V0(CUDA)");
    m.def("vadd_v1", &vadd_v1, "Vector Add V1(CUDA)");
}