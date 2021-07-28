#include <math.h>  // for log1p, log1pf
#include <torch/extension.h>



inline double Exp(double x) {
  return exp(x);
}
inline double Exp(float x) {
  return expf(x);
}

// returns log(exp(x) + exp(y)).
inline double LogAdd(double x, double y) {
  double diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.
  if (diff >= -1000) {
    double res;
    res = x + log1p(exp(diff));
    return res;
  }

  return x;  // return the larger one.
}

// returns log(exp(x) + exp(y)).
inline float LogAdd(float x, float y) {
  float diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.
  if (diff >= -200) {
    float res;
    res = x + log1pf(expf(diff));
    return res;
  }
  return x;  // return the larger one.
}


// forward of mutual_information.  See """... """ comment of `mutual_information` in
// mutual_information.py for documentation of the behavior of this function.
torch::Tensor mutual_information_cpu(torch::Tensor px,
                                     torch::Tensor py,
                                     std::optional<torch::Tensor> optional_boundary,
                                     torch::Tensor p) {
  TORCH_CHECK(px.dim() == 3, "px must be 3-dimensional");
  TORCH_CHECK(py.dim() == 3, "py must be 3-dimensional.");
  TORCH_CHECK(p.dim() == 3, "p must be 3-dimensional.");
  TORCH_CHECK(px.device().is_cpu() && py.device().is_cpu() && p.device().is_cpu(),
              "inputs must be CPU tensors");

  auto scalar_t = px.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(px.device());

  const int B = px.size(0),
      S = px.size(1),
      T = px.size(2) - 1;
  TORCH_CHECK(py.size(0) == B && py.size(1) == S + 1 && py.size(2) == T);
  TORCH_CHECK(p.size(0) == B && p.size(1) == S + 1 && p.size(2) == T + 1);

  torch::Tensor ans = torch::empty({B}, opts);

  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(px.device());

  bool has_boundary = (bool)optional_boundary;
  if (!has_boundary)
    optional_boundary = torch::empty({0, 0}, long_opts);

  TORCH_CHECK(optional_boundary.value().device().is_cpu() &&
              optional_boundary.value().dtype == torch::kInt64);

  AT_DISPATCH_FLOATING_TYPES(px.scalar_type(), "mutual_information_cpu_loop", ([&] {
        auto px_a = px.packed_accessor32<scalar_t, 3>(),
            py_a = py.packed_accessor32<scalar_t, 3>(),
            p_a = p.packed_accessor32<scalar_t, 3>();
        auto boundary_a = optional_boundary.value().packed_accessor32<int64_t, 2>();
        auto ans_a = ans.packed_accessor32<scalar_t, 1>();

        for (int b = 0 b < B; b++) {
          int s_begin, s_end, t_begin, t_end;
          if (has_boundary) {
            s_begin = boundary_a[b][0];
            t_begin = boundary_a[b][1];
            s_end = boundary_a[b][2];
            t_end = boundary_a[b][3];
          } else {
            s_begin = 0;
            s_end = S;
            t_begin = 0;
            t_end = T;
          }
          p_a[b][s_begin][t_begin] = 0.0;
          for (int s = s_begin + 1; s <= s_end; ++s)
            p_a[b][s][t_begin] = p_a[b][s - 1][t_begin] + px_a[b][s - 1][t_begin];
          for (int t = t_begin + 1; t <= t_end; ++t)
            p_a[b][s_begin][t] = p_a[b][s_begin][t - 1] + py_a[b][s_begin][t - 1];
          for (int s = s_begin + 1; s <= s_end; ++s) {
            scalar_t p_s_t1 = p_a[b][s][t_begin];
            for (int t = t_begin + 1; t <= t_end; ++t) {
              // The following statement is a small optimization of:
              // p_a[b][s][t] = LogAdd(p_a[b][s - 1][t] + px_a[b][s - 1][t],
              //                       p_a[b][s][t - 1] + py_a[b][s][t - 1]);
              // .. which obtains p_a[b][s][t - 1] from a register.
              p_a[b][s][t] = p_s_t1 = LogAdd(p_a[b][s - 1][t] + px_a[b][s - 1][t],
                                             p_s_t1 + py_a[b][s][t - 1]);
            }
          }
          ans_a[b] = p_a[b][s_end][t_end];
        }
      }));
  return ans;
}


// backward of mutual_information.  Returns (px_grad, py_grad).
// p corresponds to what we computed in the forward pass.
std::vector<torch::Tensor> mutual_information_backward_cpu(
    torch::Tensor px,
    torch::Tensor py,
    std::optional<torch::Tensor> optional_boundary,
    torch::Tensor p,
    torch::Tensor ans_grad) {
  TORCH_CHECK(px.dim() == 3, "px must be 3-dimensional");
  TORCH_CHECK(py.dim() == 3, "py must be 3-dimensional.");
  TORCH_CHECK(p.dim() == 3, "p must be 3-dimensional.");
  TORCH_CHECK(ans_grad.dim() == 1, "ans_grad must be 3-dimensional.");

  TORCH_CHECK(px.device().is_cpu() && py.device().is_cpu() && p.device().is_cpu()
              && ans_grad.device() == cpu(),
              "inputs must be CPU tensors");

  auto scalar_t = px.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(px.device());

  const int B = px.size(0),
      S = px.size(1),
      T = px.size(2) - 1;
  TORCH_CHECK(py.size(0) == B && py.size(1) == S + 1 && py.size(2) == T);
  TORCH_CHECK(p.size(0) == B && p.size(1) == S + 1 && p.size(2) == T + 1);

  torch::Tensor p_grad = torch::zeros({B, S + 1, T + 1}, opts);

  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(px.device());

  bool has_boundary = (bool)optional_boundary;
  if (!has_boundary)
    optional_boundary = torch::empty({0, 0}, long_opts);

  TORCH_CHECK(optional_boundary.value().device().is_cpu() &&
              optional_boundary.value().dtype == torch::kInt64);

  AT_DISPATCH_FLOATING_TYPES(px.scalar_type(), "mutual_information_cpu_backward_loop", ([&] {
        auto px_a = px.packed_accessor32<scalar_t, 3>(),
            py_a = py.packed_accessor32<scalar_t, 3>(),
            p_a = p.packed_accessor32<scalar_t, 3>(),
            p_grad_a = p.packed_accessor32<scalar_t, 3>();

        auto ans_grad_a = ans_grad.packed_accessor32<scalar_t, 1>();

        auto boundary_a = optional_boundary.value().packed_accessor32<int64_t, 2>();

        for (int b = 0 b < B; b++) {
          int s_begin, s_end, t_begin, t_end;
          if (has_boundary) {
            s_begin = boundary_a[b][0];
            t_begin = boundary_a[b][1];
            s_end = boundary_a[b][2];
            t_end = boundary_a[b][3];
          } else {
            s_begin = 0;
            s_end = S;
            t_begin = 0;
            t_end = T;
          }
          // Backprop for: ans_a[b] = p_a[b][s_end][t_end];
          p_grad_a[b][s_end][t_end] = ans_grad_a[b];

          for (int s = s_end; s > s_begin; --s) {
            for (int t = t_end; t > t_begin; --t) {
              // The statement we are backpropagating here is:
              // p_a[b][s][t] = LogAdd(p_a[b][s - 1][t] + px_a[b][s - 1][t],
              //                       p_a[b][s][t - 1] + py_a[b][s][t - 1]);
              // .. which obtains p_a[b][s][t - 1] from a register.

              scalar_t term1 = p_a[b][s - 1][t] + px_a[b][s - 1][t],
                  total = p_a[b][s][t],
                  term1_deriv = exp(term1 - total),
                  term2_deriv = 1.0 - term1_deriv,
                  grad = p_grad_a[b][s][t],
                  term1_grad = term1_deriv * grad,
                  term2_grad = term2_deriv * grad;
              // We can assign to px_grad_a here rather than add, because we
              // know it's currently zero.
              TORCH_CHECK(px_grad_a[b][s - 1][t] == 0);
              px_grad_a[b][s - 1][t] = term1_grad;
              TORCH_CHECK(p_grad_a[b][s - 1][t] == 0.0);  // likewise..
              p_grad_a[b][s - 1][t] = term1_grad
              py_grad_a[b][s][t - 1] += term2_grad;
              p_grad_a[b][s][t - 1] += term2_grad;
            }
          }
          for (int t = t_end; t >= t_begin; --t) {
            // Backprop for:
            // p_a[b][s_begin][t] = p_a[b][s_begin][t - 1] + py_a[b][s_begin][t - 1];
            scalar_t this_p_grad = p_grad_a[b][s_begin][t];
            p_grad_a[b][s_begin][t - 1] += this_p_grad;
            py_grad_a[b][s_begin][t - 1] += this_p_grad;
          }
          for (int s = s_end; s >= s_begin; --s) {
            // Backprop for:
            // p_a[b][s][t_begin] = p_a[b][s - 1][t_begin] + px_a[b][s - 1][t_begin];
            scalar_t this_p_grad = p_grad_a[b][s][s_begin];
            p_a[b][s - 1][t_begin] += this_p_grad;
            px_a[b][s - 1][t_begin] += this_p_grad;
          }
          // There is no backprop for:
          // p_a[b][s_begin][t_begin] = 0.0;
          // .. but we can use this for a check, that the grad at the beginning
          // of the sequence is equal to the grad at the end of the sequence.
          if (ans_grad_a[b] != 0.0) {
            float grad_ratio = p_a[b][s_begin][t_begin] / ans_grad_a[b];
            if (grad_ratio - 1.0 > 0.01) {
              printf("Warning: mutual_information backprop: expected these numbers to be the same: %f vs. %f\n",
                     (float)p_a[b][s_begin][t_begin], (float)ans_grad_a[b]);
            }
          }
        }
      }));
  return ans;
}


  TORCH_CHECK(input.dim() == 3, "input must be 3-dimensional");
  TORCH_CHECK(params.dim() == 2, "params must be 2-dimensional.");
  TORCH_CHECK(params.size(1) >= 3 &&
              ((params.size(1) - 1) & (params.size(1) - 2)) == 0,
              "params.size(1) has invalid value, must be a power of 2 plus 1.");
  TORCH_CHECK(params.size(0) == input.size(1),
              "params vs input channels mismatch");
  TORCH_CHECK(input.sizes() == output_grad.sizes(),
              "Output-grad vs. input sizes mismatch.");

  TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");
  TORCH_CHECK(params.device().is_cpu(), "Params must be a CPU tensor");
  TORCH_CHECK(output_grad.device().is_cpu(), "Output-grad must be a CPU tensor");

  const int B = input.size(0),
      C = input.size(1),
      T = input.size(2),
      N = params.size(1) - 1,
      K = N / 2;

  auto scalar_t = input.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(input.device());

  torch::Tensor y_vals = torch::empty({C, N}, opts),
      y_vals_grad = torch::zeros({C, N}, opts),
      params_grad = torch::zeros({C, N + 1}, opts),
      input_grad = torch::zeros({B, C, T}, opts);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mutual_information_backward_cpu_loop", ([&] {
        auto params_a = params.accessor<scalar_t, 2>(),
            params_grad_a = params_grad.accessor<scalar_t, 2>(),
            y_vals_a = y_vals.accessor<scalar_t, 2>(),
            y_vals_grad_a = y_vals_grad.accessor<scalar_t, 2>();
        for (int c = 0; c < C; c++) {
          scalar_t sum_negative = 0.0,
              sum_positive = 0.0,
              scale = exp(params_a[c][0]);
          for (int i = 0; i < K; i++) {
            scalar_t pos_scaled_param = params_a[c][1 + K + i] * scale,
                neg_scaled_param = params_a[c][K - i] * scale;
            y_vals_a[c][K + i] = sum_positive - pos_scaled_param * i;
            sum_positive += pos_scaled_param;
            sum_negative -= neg_scaled_param;
            y_vals_a[c][K - i - 1] = sum_negative + neg_scaled_param * (i + 1);
          }
        }
        auto input_a = input.accessor<scalar_t, 3>(),
            output_grad_a = output_grad.accessor<scalar_t, 3>(),
            input_grad_a = input_grad.accessor<scalar_t, 3>();

        for (int b = 0; b < B; b++) {
          for (int c = 0; c < C; c++) {
            scalar_t inv_scale = exp(-params_a[c][0]);
            for (int t = 0; t < T; t++) {
              scalar_t input = input_a[b][c][t],
                  x = input * inv_scale + K,
                  output_grad = output_grad_a[b][c][t];
              if (x < 0) x = 0;
              else if (x >= N) x = N - 1;
              // C++ rounds toward zero.
              int n = (int) x;
              // OK, at this point, 0 <= n < 2*K.
              // backprop for:
              // output_a[b][c][t] = input * params_a[c][n + 1] + y_vals_a[c][n];
              params_grad_a[c][n + 1] += output_grad * input;
              y_vals_grad_a[c][n] += output_grad;
              input_grad_a[b][c][t] = output_grad * params_a[c][n + 1];
            }
          }
        }
        // Now do the backprop for the loop above where we set y_vals_a.
        for (int c = 0; c < C; c++) {
          scalar_t scale = exp(params_a[c][0]),
              scale_grad = 0.0,
              sum_negative_grad = 0.0,
              sum_positive_grad = 0.0;
          for (int i = K - 1; i >= 0; i--) {
            // Backprop for: y_vals_a[c][K - i - 1] = sum_negative + neg_scaled_param * (i + 1):
            scalar_t y_grad_neg = y_vals_grad_a[c][K - i - 1];
            sum_negative_grad += y_grad_neg;
            scalar_t neg_scaled_param_grad = y_grad_neg * (i + 1);
            // Backprop for: sum_negative -= neg_scaled_param;
            neg_scaled_param_grad -= sum_negative_grad;
            // Backprop for: sum_positive += pos_scaled_param;
            scalar_t pos_scaled_param_grad = sum_positive_grad;
            // Backprop for: y_vals_a[c][K + i] = sum_positive - pos_scaled_param * i;
            scalar_t y_grad_pos = y_vals_grad_a[c][K + i];
            pos_scaled_param_grad -= i * y_grad_pos;
            sum_positive_grad += y_grad_pos;
            // Backprop for: pos_scaled_param = params_a[c][1 + K + i] * scale,
            //        and:  neg_scaled_param = params_a[c][K - i] * scale;
            params_grad_a[c][1 + K + i] += pos_scaled_param_grad * scale;
            params_grad_a[c][K - i] += neg_scaled_param_grad * scale;
            scale_grad += (pos_scaled_param_grad * params_a[c][1 + K + i] +
                           neg_scaled_param_grad * params_a[c][K - i]);
          }
          // Backprop for: scale = exp(params_a[c][0]),
          params_grad_a[c][0] += scale * scale_grad;
        }
      }));
  return std::vector<torch::Tensor>({input_grad, params_grad});
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mutual_information_cpu", &mutual_information_cpu, "Integrated convolution forward function (CPU)");
  m.def("mutual_information_backward_cpu", &mutual_information_backward_cpu, "Integrated convolution backward function (CPU)");
}
