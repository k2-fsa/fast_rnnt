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
  dtype scalar_t = input.dtype();
  TORCH_CHECK(pos_add.dtype() == scalar_t &&
              pos_mul.dtype() == scalar_t,
              "Input dtypes mismatch");

  torch::Tensor output = torch::empty({N, C, H, W},
                                      torch::TensorOptions().dtype(scalar_t).device(input.device()));

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "integrated_conv_cpu_loop", ([&] {
        auto input_a = input.accessor<scalar_t, 4>(),
            pos_add_a = pos_add.accessor<scalar_t, 3>(),
            pos_mul_a = pos_add.accessor<scalar_t, 3>(),
            output_a = pos_add.accessor<scalar_t, 4>();
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
                    int src_w = h + kh - kH / 2;
                    scalar_t src = 0.0;
                    if (static_cast<unsigned int>(src_h) < static_cast<unsigned int>(H) &&
                        static_cast<unsigned int>(src_w) < static_cast<unsigned int>(W))
                      src = src_input_a[src_h][src_w];
                    scalar_t relu = src + dest + this_pos_add_a;
                    if (relu > 0.0)
                      sum += relu * this_pos_mul_a;
                  }
                }
                output_a[h][w] = sum;
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
  // TODO.
  return std::vector<torch::Tensor>();
}



template <typename T_accessor, typename scalar_t>
inline
void discounted_sum_update(T_accessor &accessor, int batchsz, scalar_t gamma, int change_pos, int discounted_pos) {
    for (int i=0; i<batchsz-3; i+=4) {
        accessor[i+0][change_pos] += gamma * accessor[i+0][discounted_pos];
        accessor[i+1][change_pos] += gamma * accessor[i+1][discounted_pos];
        accessor[i+2][change_pos] += gamma * accessor[i+2][discounted_pos];
        accessor[i+3][change_pos] += gamma * accessor[i+3][discounted_pos];
    }
    for (int i=(batchsz - (batchsz & 3)); i<batchsz; i++) {
        accessor[i][change_pos] += gamma * accessor[i][discounted_pos];
    }
}


torch::Tensor discounted_cumsum_left_cpu(torch::Tensor x, double gamma) {
    TORCH_CHECK(x.device().is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(0.0 <= gamma && gamma <= 1.0, "Gamma must be in the range [0,1]");

    auto y = x.clone();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "discounted_cumsum_left_cpu_loop", ([&] {
        auto ya = y.accessor<scalar_t, 2>();
        for (int j=0; j<y.size(1); j++) {
            int j_left = j-1;
            if (j_left == -1) {
                continue;
            }
            discounted_sum_update(ya, y.size(0), gamma, j, j_left);
        }
    }));

    return y;
}


torch::Tensor discounted_cumsum_right_cpu(torch::Tensor x, double gamma) {
  TORCH_CHECK(x.device().is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(0.0 <= gamma && gamma <= 1.0, "Gamma must be in the range [0,1]");

    auto y = x.clone();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "discounted_cumsum_right_cpu_loop", ([&] {
        auto ya = y.accessor<scalar_t, 2>();
        for (int j=y.size(1)-1; j>=0; j--) {
            int j_right = j+1;
            if (j_right == y.size(1)) {
                continue;
            }
            discounted_sum_update(ya, y.size(0), gamma, j, j_right);
        }
    }));

    return y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("integrated_conv_cpu", &integrated_conv_cpu, "Integrated convolution forward function (CPU)");
  m.def("integrated_conv_backward_cpu", &integrated_conv_forward_cpu, "Integrated convolution backward function (CPU)");
}
