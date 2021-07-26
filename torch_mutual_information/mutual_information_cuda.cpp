#include <torch/extension.h>


// forward of mutual_information.  """... """ comment of `mutual_information`
// in mutual_information.py documents the behavior of this function.
torch::Tensor mutual_information_cuda(torch::Tensor input,
                                  torch::Tensor params);


// backward of mutual_information; returns (grad_input, grad_params).
std::vector<torch::Tensor> mutual_information_backward_cuda(torch::Tensor input,
                                                        torch::Tensor params,
                                                        torch::Tensor grad_output);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mutual_information_cuda", &mutual_information_cuda, "Learned nonlinearity forward function (CUDA)");
  m.def("mutual_information_backward_cuda", &mutual_information_backward_cuda, "Learned nonlinearity backward function (CUDA)");
}
