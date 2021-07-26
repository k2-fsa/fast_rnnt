#include <math.h>  // for log1p, log1pf
#include <torch/extension.h>



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

  if (diff >= kMinLogDiffDouble) {
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

  if (diff >= kMinLogDiffFloat) {
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
                                     torch::Tensor q) {

  TORCH_CHECK(px.dim() == 3, "px must be 3-dimensional");
  TORCH_CHECK(py.dim() == 3, "params must be 3-dimensional.");
  TORCH_CHECK(q.dim() == 3, "params must be 3-dimensional.");

  auto scalar_t = px.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(px.device());


  const int B = px.size(0),
      S = px.size(1),
      T = px.size(2);

  TORCH_CHECK(q.size(0) == B && q.size(1) == S + T && q.size(2) == T);


  auto long_opts = torch::TensorOptiona().dtype(torch::kInt64);

  bool has_boundary = (bool)optional_boundary;
  if (!has_boundary)
    optional_boundary = torch::empty({}, long_opts);




  AT_DISPATCH_FLOATING_TYPES(px.scalar_type(), "mutual_information_cpu_loop", ([&] {
        auto px_a = px.accessor<scalar_t, 3>(),
            py_a = py.accessor<scalar_t, 3>();
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
            output_a = output.accessor<scalar_t, 3>();

        for (int b = 0; b < B; b++) {
          for (int c = 0; c < C; c++) {
            scalar_t scale = exp(params_a[c][0]),
                inv_scale = 1.0 / scale;
            for (int t = 0; t < T; t++) {
              // `x` is the scaled input x plus an offset so that -K maps to 0.
              //  Note: the discontinuities in our function are at -(K-1) ... +(K+1),
              // so in a sense -K and +K are not special, but we include those
              // extra values as an easy way to handle the semi-infinite regions
              // that are < -(K-1) and > (K-1)
              scalar_t input = input_a[b][c][t],
                  x = input * inv_scale + K;
              if (x < 0) x = 0;
              else if (x >= N) x = N - 1;
              // C++ rounds toward zero.
              int n = (int) x;
              // OK, at this point, 0 <= min < 2*K.
              output_a[b][c][t] = input * params_a[c][n + 1] + y_vals_a[c][n];
            }
          }
        }}));
  return output;
}


// backward of mutual_information.  Returns (input_grad, params_grad)

std::vector<torch::Tensor> mutual_information_backward_cpu(torch::Tensor input,
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
