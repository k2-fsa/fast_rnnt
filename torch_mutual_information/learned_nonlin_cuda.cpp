#include <torch/extension.h>


// forward of learned_nonlin.  """... """ comment of `learned_nonlin`
// in learned_nonlin.py documents the behavior of this function.
torch::Tensor learned_nonlin_cuda(torch::Tensor input,
                                  torch::Tensor params);


// backward of learned_nonlin; returns (grad_input, grad_params).
std::vector<torch::Tensor> learned_nonlin_backward_cuda(torch::Tensor input,
                                                        torch::Tensor params,
                                                        torch::Tensor grad_output);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("learned_nonlin_cuda", &learned_nonlin_cuda, "Learned nonlinearity forward function (CUDA)");
  m.def("learned_nonlin_backward_cuda", &learned_nonlin_backward_cuda, "Learned nonlinearity backward function (CUDA)");
}
