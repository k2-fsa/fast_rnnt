#include <torch/extension.h>


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
    m.def("discounted_cumsum_left_cpu", &discounted_cumsum_left_cpu, "Discounted Cumulative Sum CPU (Left)");
    m.def("discounted_cumsum_right_cpu", &discounted_cumsum_right_cpu, "Discounted Cumulative Sum CPU (Right)");
}
