#include <torch/extension.h>


torch::Tensor discounted_cumsum_right(torch::Tensor x, double gamma);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("discounted_cumsum_right", &discounted_cumsum_right,
      "Discounted Cumulative Sum Right");
}
