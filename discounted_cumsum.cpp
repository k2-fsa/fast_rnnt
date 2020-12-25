#include <torch/extension.h>

torch::Tensor discounted_cumsum_right_minthreads(torch::Tensor x, double gamma);
torch::Tensor discounted_cumsum_right_coalesced(torch::Tensor x, double gamma);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("discounted_cumsum_right_minthreads", &discounted_cumsum_right_minthreads,
      "Discounted Cumulative Sum Right Minimum Threads");
  m.def("discounted_cumsum_right_coalesced", &discounted_cumsum_right_coalesced,
      "Discounted Cumulative Sum Right Coalesced Writes");
}
