#include <torch/extension.h>


// forward of integrated_conv.  """... """ comment of `integrated_conv`
// in integrated_conv.py documents the behavior of this function.
torch::Tensor integrated_conv_cuda(torch::Tensor input,
                                   torch::Tensor pos_add,
                                   torch::Tensor pos_mul);

// backward of integrated_conv; returns (grad_input, grad_pos_add, grad_pos_mul).
std::vector<torch::Tensor> integrated_conv_backward_cuda(torch::Tensor input,
                                                         torch::Tensor pos_add,
                                                         torch::Tensor pos_mul,
                                                         torch::Tensor grad_output);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("integrated_conv_cuda", &integrated_conv_cuda, "Integrated convolution forward function (CUDA)");
  m.def("integrated_conv_backward_cuda", &integrated_conv_forward_cuda, "Integrated convolution backward function (CUDA)");
}
