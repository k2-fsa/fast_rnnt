#include <torch/extension.h>


torch::Tensor discounted_cumsum_left(torch::Tensor x, double gamma);
torch::Tensor discounted_cumsum_right(torch::Tensor x, double gamma);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("discounted_cumsum_left", &discounted_cumsum_left, "Discounted Cumulative Sum (Left)");
    m.def("discounted_cumsum_right", &discounted_cumsum_right, "Discounted Cumulative Sum (Right)");
}
