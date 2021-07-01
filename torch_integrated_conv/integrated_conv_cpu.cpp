#include <torch/extension.h>



// forward of integrated_conv.  """... """ comment of `integrated_conv`
// in integrated_conv.py documents the behavior of this function.
torch::Tensor integrated_conv_cpu(torch::Tensor input,
                                  torch::Tensor pos_add,
                                  torch::Tensor pos_mul) {
  TORCH_CHECK(input.dim() == 4, "input must be 4-dimensional");
  TORCH_CHECK(pos_add.dim() == 3, "pos_add must be 3-dimensional.");
  TORCH_CHECK(pos_mul.dim() == 3, "pos_add must be 3-dimensional.");
  TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");
  const int N = input.size(0),
      C = input.size(1) / 2,
      H = input.size(2),
      W = input.size(3),
      kH = pos_add.size(1),
      kW = pos_add.size(2);
  TORCH_CHECK(kH % 2 == 1 && kW % 2 == 1);
  TORCH_CHECK(input.size(1) % 2 == 0, "Input must have even num-channels");
  TORCH_CHECK(pos_add.size(0) == C && pos_mul.size(0) == C &&
              pos_mul.size(1) == kH && pos_mul.size(2) == kW,
              "Input sizes mismatch.");
  TORCH_CHECK(pos_add.device() == input.device() &&
              pos_mul.device() == pos_add.device(),
              "Input devices mismatch");
  auto scalar_t = input.scalar_type();
  TORCH_CHECK(pos_add.scalar_type() == scalar_t &&
              pos_mul.scalar_type() == scalar_t,
              "Input dtypes mismatch");

  torch::Tensor output = torch::empty({N, C, H, W},
                                      torch::TensorOptions().dtype(scalar_t).device(input.device()));

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "integrated_conv_cpu_loop", ([&] {
        auto input_a = input.accessor<scalar_t, 4>(),
            output_a = output.accessor<scalar_t, 4>();
        auto pos_add_a = pos_add.accessor<scalar_t, 3>(),
            pos_mul_a = pos_mul.accessor<scalar_t, 3>();

        for (int n = 0; n < N; n++) {
          for (int c = 0; c < C; c++) {
            auto src_input_a = input_a[n][c],
                this_pos_add_a = pos_add_a[c],
                this_pos_mul_a = pos_mul_a[c],
                this_output_a = output_a[n][c];
            for (int h = 0; h < H; h++) {
              for (int w = 0; w < W; w++) {
                scalar_t dest = input_a[n][c + C][h][w],
                    sum = 0.0;
                for (int kh = 0; kh < kH; kh++) {
                  int src_h = h + kh - kH / 2;
                  for (int kw = 0; kw < kW; kw++) {
                    int src_w = w + kw - kW / 2;
                    scalar_t src = 0.0;
                    if (static_cast<unsigned int>(src_h) < static_cast<unsigned int>(H) &&
                        static_cast<unsigned int>(src_w) < static_cast<unsigned int>(W))
                      src = src_input_a[src_h][src_w];
                    scalar_t relu = src + dest + this_pos_add_a[kh][kw];
                    if (relu > 0.0)
                      sum += relu * this_pos_mul_a[kh][kw];
                  }
                }
                this_output_a[h][w] = sum;
              }
            }
          }
        }
      }));
  return output;
}

// backward of integrated_conv; returns (grad_input, grad_pos_add, grad_pos_mul).
std::vector<torch::Tensor> integrated_conv_backward_cpu(torch::Tensor input,
                                                        torch::Tensor pos_add,
                                                        torch::Tensor pos_mul,
                                                        torch::Tensor grad_output) {
  TORCH_CHECK(input.dim() == 4, "input must be 4-dimensional");
  TORCH_CHECK(pos_add.dim() == 3, "pos_add must be 3-dimensional.");
  TORCH_CHECK(pos_mul.dim() == 3, "pos_add must be 3-dimensional.");
  TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");
  const int N = input.size(0),
      C = input.size(1) / 2,
      H = input.size(2),
      W = input.size(3),
      kH = pos_add.size(1),
      kW = pos_add.size(2);
  TORCH_CHECK(kH % 2 == 1 && kW % 2 == 1);
  TORCH_CHECK(input.size(1) % 2 == 0, "Input must have even num-channels");
  TORCH_CHECK(pos_add.size(0) == C && pos_mul.size(0) == C &&
              pos_mul.size(1) == kH && pos_mul.size(2) == kW,
              "Input sizes mismatch.");
  TORCH_CHECK(pos_add.device() == input.device() &&
              pos_mul.device() == pos_add.device(),
              "Input devices mismatch");
  auto scalar_t = input.scalar_type();
  TORCH_CHECK(pos_add.scalar_type() == scalar_t &&
              pos_mul.scalar_type() == scalar_t,
              "Input dtypes mismatch");
  TORCH_CHECK(grad_output.dim() == 4 && grad_output.size(0) == N
              && grad_output.size(1) == C && grad_output.size(2) == H
              && grad_output.size(3) == W);



  torch::Tensor grad_input = torch::zeros({N, 2*C, H, W},
                                          torch::TensorOptions().dtype(scalar_t).device(input.device())),
      grad_pos_add = torch::zeros({C, kH, kW},
                                  torch::TensorOptions().dtype(scalar_t).device(input.device())),
      grad_pos_mul = torch::zeros({C, kH, kW},
                                  torch::TensorOptions().dtype(scalar_t).device(input.device()));


  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "integrated_conv_cpu_loop", ([&] {
        auto input_a = input.accessor<scalar_t, 4>(),
            grad_output_a = grad_output.accessor<scalar_t, 4>(),
            grad_input_a = grad_input.accessor<scalar_t, 4>();
        auto pos_add_a = pos_add.accessor<scalar_t, 3>(),
            pos_mul_a = pos_mul.accessor<scalar_t, 3>(),
            grad_pos_add_a = grad_pos_add.accessor<scalar_t, 3>(),
            grad_pos_mul_a = grad_pos_mul.accessor<scalar_t, 3>();

        for (int n = 0; n < N; n++) {
          for (int c = 0; c < C; c++) {

            for (int h = 0; h < H; h++) {
              for (int w = 0; w < W; w++) {
                scalar_t dest = input_a[n][c + C][h][w],
                    dest_grad = 0.0,  // to be multiplied by this_output_grad later..
                    this_grad_output = grad_output_a[n][c][h][w];
                for (int kh = 0; kh < kH; kh++) {
                  int src_h = h + kh - kH / 2;
                  for (int kw = 0; kw < kW; kw++) {
                    int src_w = w + kw - kW / 2;
                    scalar_t src = 0.0;
                    if (static_cast<unsigned int>(src_h) < static_cast<unsigned int>(H) &&
                        static_cast<unsigned int>(src_w) < static_cast<unsigned int>(W))
                      src = input_a[n][c][src_h][src_w];
                    scalar_t relu = src + dest + pos_add_a[c][kh][kw];
                    if (relu >= 0.0) {
                      scalar_t pos_mul_val = pos_mul_a[c][kh][kw];
                      dest_grad += pos_mul_val;  // will later multiply by this_output_grad
                      grad_pos_add_a[c][kh][kw] += this_grad_output * pos_mul_val;
                      grad_pos_mul_a[c][kh][kw] += this_grad_output * relu;
                      if (static_cast<unsigned int>(src_h) < static_cast<unsigned int>(H) &&
                          static_cast<unsigned int>(src_w) < static_cast<unsigned int>(W))
                        grad_input_a[n][c][src_h][src_w] += this_grad_output * pos_mul_val;
                    }
                  }
                }
                grad_input_a[n][c + C][h][w] += dest_grad * this_grad_output;
              }
            }
          }
        }
      }));

  return std::vector<torch::Tensor>({grad_input, grad_pos_add, grad_pos_mul});
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("integrated_conv_cpu", &integrated_conv_cpu, "Integrated convolution forward function (CPU)");
  m.def("integrated_conv_backward_cpu", &integrated_conv_backward_cpu, "Integrated convolution backward function (CPU)");
}
