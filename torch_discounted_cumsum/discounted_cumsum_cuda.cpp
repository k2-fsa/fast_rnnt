#include <torch/extension.h>


torch::Tensor discounted_cumsum_left_cuda(torch::Tensor x, double gamma);
torch::Tensor discounted_cumsum_right_cuda(torch::Tensor x, double gamma);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("discounted_cumsum_left_cuda", &discounted_cumsum_left_cuda, "Discounted Cumulative Sum CUDA (Left)");
    m.def("discounted_cumsum_right_cuda", &discounted_cumsum_right_cuda, "Discounted Cumulative Sum CUDA (Right)");
}
