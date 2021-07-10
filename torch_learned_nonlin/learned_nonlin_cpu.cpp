#include <torch/extension.h>



// forward of learned_nonlin.  See """... """ comment of `learned_nonlin` in
// learned_nonlin.py for documentation of the behavior of this function.
torch::Tensor learned_nonlin_cpu(torch::Tensor input,
                                 torch::Tensor params) {
  TORCH_CHECK(input.dim() == 3, "input must be 3-dimensional");
  TORCH_CHECK(params.dim() == 2, "params must be 2-dimensional.");
  TORCH_CHECK(params.size(1) >= 3 &&
              ((params.size(1) - 1) & (params.size(1) - 2)) == 0,
              "params.size(1) has invalid value, must be a power of 2 plus 1.");
  TORCH_CHECK(params.size(0) == input.size(1),
              "params vs input channels mismatch");

  TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");
  TORCH_CHECK(params.device().is_cpu(), "Params must be a CPU tensor");

  const int B = input.size(0),
      C = input.size(1),
      T = input.size(2),
      N = params.size(1) - 1,
      K = N / 2;

  auto scalar_t = input.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(input.device());

  torch::Tensor y_vals = torch::empty({C, N}, opts),
    output = torch::empty({B, C, T}, opts);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "learned_nonlin_cpu_loop", ([&] {
        auto params_a = params.accessor<scalar_t, 2>(),
            y_vals_a = y_vals.accessor<scalar_t, 2>();
        for (int c = 0; c < C; c++) {
          scalar_t sum_negative = 0.0,
              sum_positive = 0.0,
              scale = exp(params_a[c][0]);

          for (int i = 0; i < K; i++) {
            y_vals_a[c][K + i] = sum_positive * scale;
            y_vals_a[c][K - i] = sum_negative * scale;
            sum_positive += params_a[c][1 + K + i];
            sum_negative -= params_a[c][K - i];
          }
          // the reference point for the lowest, half-infinite interval (the one
          // starting at x=-(K-1) is still x=-(K-1); this value is repeated in y_vals.
          y_vals_a[c][0] = y_vals_a[c][1];
        }

        auto input_a = input.accessor<scalar_t, 3>(),
            output_a = output.accessor<scalar_t, 3>();

        for (int b = 0; b < B; b++) {
          for (int c = 0; c < C; c++) {
            scalar_t inv_scale = exp(-params_a[c][0]);
            for (int t = 0; t < T; t++) {
              // `x` is the scaled input x plus an offset so that -K maps to 0.
              //  Note: the discontinuities in our function are at -(K-1) ... +(K+1),
              // so in a sense -K and +K are not special, but we include those
              // extra values as an easy way to handle the semi-infinite regions
              // that are < -(K-1) and > (K-1)
              scalar_t x = input_a[b][c][t] * inv_scale + K;
              int min = 0, diff = K;
              while (diff > 0) {
                int mid = min + diff;
                if (x >= mid)
                  min = mid;
                diff = diff >> 1;
              }
              // OK, at this point, 0 <= min < 2*K.
              scalar_t y = (x - (scalar_t)min) * params_a[c][min + 1] + y_vals_a[c][min];
              // printf("x = %f, y = %f, min = %d; y = (%f - %d) * %f+ %f\n", x, y, min,
              // x, min, params_a[c][min + 1], y_vals_a[c][min - 1]);
              output_a[b][c][t] = y;
            }
          }
        }}));
  return output;
}


// backward of learned_nonlin.  Returns (input_grad, params_grad)

std::vector<torch::Tensor> learned_nonlin_backward_cpu(torch::Tensor input,
                                                       torch::Tensor params,
                                                       torch::Tensor output_grad) {
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

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "learned_nonlin_backward_cpu_loop", ([&] {
        auto params_a = params.accessor<scalar_t, 2>(),
            params_grad_a = params_grad.accessor<scalar_t, 2>(),
            y_vals_a = y_vals.accessor<scalar_t, 2>(),
            y_vals_grad_a = y_vals_grad.accessor<scalar_t, 2>();
        for (int c = 0; c < C; c++) {
          scalar_t sum_negative = 0.0,
              sum_positive = 0.0,
              scale = exp(params_a[c][0]);
          for (int i = 0; i < K; i++) {
            y_vals_a[c][K + i] = sum_positive;
            y_vals_a[c][K - i] = sum_negative;
            sum_positive += params_a[c][1 + K + i] * scale;
            sum_negative -= params_a[c][K - i] * scale;
          }
          // the reference point for the lowest, half-infinite interval (the one
          // starting at x=-(K-1) is still x=-(K-1); this value is repeated in y_vals.
          y_vals_a[c][0] = y_vals_a[c][1];
        }

        auto input_a = input.accessor<scalar_t, 3>(),
            output_grad_a = output_grad.accessor<scalar_t, 3>(),
            input_grad_a = input_grad.accessor<scalar_t, 3>();

        for (int b = 0; b < B; b++) {
          for (int c = 0; c < C; c++) {
            scalar_t inv_scale = exp(-params_a[c][0]),
                inv_scale_grad = 0.0;
            for (int t = 0; t < T; t++) {
              // `x` is the scaled input x plus an offset so that -K maps to 0.
              //  Note: the discontinuities in our function are at -(K-1) ... +(K+1),
              // so in a sense -K and +K are not special, but we include those
              // extra values as an easy way to handle the semi-infinite regions
              // that are < -(K-1) and > (K-1)
              scalar_t input = input_a[b][c][t],
                  x = input * inv_scale + K,
                  y_grad = output_grad_a[b][c][t];
              int min = 0, diff = K;
              while (diff > 0) {
                int mid = min + diff;
                if (x >= mid)
                  min = mid;
                diff = diff >> 1;
              }
              // OK, at this point, 0 <= min < 2*K.
              // backprop for:
              // scalar_t y = (x - (scalar_t)min) * params_a[c][min + 1] + y_vals_a[c][min];
              scalar_t x_grad = y_grad * params_a[c][min + 1];
              params_grad_a[c][min + 1] += y_grad * (x - (scalar_t)min);
              y_vals_grad_a[c][min] += y_grad;
              // backprop for:  x = input * inv_scale + K,
              inv_scale_grad += x_grad * input;
              input_grad_a[b][c][t] = x_grad * inv_scale;
            }
            // Do the backprop for: inv_scale = exp(-params_a[c][0])
            params_grad_a[c][0] -= inv_scale * inv_scale_grad;
          }
        }
        // Now do the backprop for the loop above where we set y_vals_a.
        for (int c = 0; c < C; c++) {
          // backprop for: y_vals_a[c][0] = y_vals_a[c][1];
          y_vals_grad_a[c][1] += y_vals_grad_a[c][0];
          scalar_t scale = exp(params_a[c][0]),
              inv_scale = 1.0 / scale,
              scale_grad = 0.0,
              sum_negative_grad = 0.0,
              sum_positive_grad = 0.0;
          for (int i = K - 1; i >= 0; i--) {
            // backprop for: sum_negative -= params_a[c][K - i];
            params_grad_a[c][K - i] -= sum_negative_grad;
            // backprop for: sum_positive += params_a[c][1 + K + i] * scale;
            params_grad_a[c][1 + K + i] += sum_positive_grad;
            // backprop for:  y_vals_a[c][K - i] = sum_negative * scale;
            sum_negative_grad += y_vals_grad_a[c][K - i] * scale;
            // The next code line is equivalent to:
            //  scale_grad += y_vals_grad_a[c][K - i] * sum_negative, substituting:
            //  sum_negative == y_vals_a[c][K - i] / scale
            scale_grad += y_vals_grad_a[c][K - i] * y_vals_a[c][K - i] * inv_scale;
            // backprop for: y_vals_a[c][K + i] = sum_positive * scale;
            sum_positive_grad += y_vals_grad_a[c][K + i] * scale;
            scale_grad += y_vals_grad_a[c][K + i] * y_vals_a[c][K + i] * inv_scale;
          }
          // Backprop for: scale = exp(params_a[c][0]),
          params_grad_a[c][0] += scale * scale_grad;
        }
      }));
  return std::vector<torch::Tensor>({input_grad, params_grad});
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("learned_nonlin_cpu", &learned_nonlin_cpu, "Integrated convolution forward function (CPU)");
  m.def("learned_nonlin_backward_cpu", &learned_nonlin_backward_cpu, "Integrated convolution backward function (CPU)");
}
