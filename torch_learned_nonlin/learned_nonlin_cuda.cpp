#include <torch/extension.h>


// forward of learned_nonlin.  """... """ comment of `learned_nonlin`
// in learned_nonlin.py documents the behavior of this function.
torch::Tensor learned_nonlin_cuda(torch::Tensor input,
                                   torch::Tensor pos_add,
                                   torch::Tensor pos_mul);

// backward of learned_nonlin; returns (grad_input, grad_pos_add, grad_pos_mul).
std::vector<torch::Tensor> learned_nonlin_backward_cuda(torch::Tensor input,
                                                         torch::Tensor pos_add,
                                                         torch::Tensor pos_mul,
                                                         torch::Tensor grad_output);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("learned_nonlin_cuda", &learned_nonlin_cuda, "Integrated convolution forward function (CUDA)");
  m.def("learned_nonlin_backward_cuda", &learned_nonlin_backward_cuda, "Integrated convolution backward function (CUDA)");
}
