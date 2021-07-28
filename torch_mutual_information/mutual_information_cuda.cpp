#include <torch/extension.h>


// forward of mutual_information.  """... """ comment of `mutual_information`
// in mutual_information.py documents the behavior of this function.
// It is the core recursion in the sequence-to-sequence mutual information
// computation.
// returns 'ans', of dimension B (batch size).
torch::Tensor mutual_information_cuda(torch::Tensor px,  // [B][S][T+1]
                                      torch::Tensor py,  // [B][S+1][T]
                                      std::optional<torch::Tensor> boundary_info,  // [B][4], int64_t.
                                      torch::Tensor p);  //  [B][S+1][T+1]; an output


// backward of mutual_information; returns (grad_px, grad_py)
std::vector<torch::Tensor> mutual_information_backward_cuda(
    torch::Tensor px,
    torch::Tensor py,
    std::optional<torch::Tensor> boundary_info,
    torch::Tensor p,
    torch::Tensor ans_grad);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mutual_information_cuda", &mutual_information_cuda, "Mutual information forward function (CUDA)");
  m.def("mutual_information_backward_cuda", &mutual_information_backward_cuda, "Mutual information backward function (CUDA)");
}
