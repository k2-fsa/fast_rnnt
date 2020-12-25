#include <torch/extension.h>


template <typename scalar_t>
__device__ __forceinline__
scalar_t discounted_sum_power(scalar_t a, scalar_t b, scalar_t gamma, int power) {
    return a + b * pow(gamma, scalar_t(power));
}


enum SumDirection {
    SUM_DIRECTION_LEFT,
    SUM_DIRECTION_RIGHT,
};


template <SumDirection sum_direction>
__device__ __forceinline__
void resolve_positions(
    const int &stride_prev_group, const int &stride_cur_group, const int &group_of_thread, const int &thread_in_group,
    int &change_pos, int &discounted_pos, int &discount_power
);


template <>
__device__ __forceinline__
void resolve_positions<SUM_DIRECTION_LEFT>(
    const int &stride_prev_group, const int &stride_cur_group, const int &group_of_thread, const int &thread_in_group,
    int &change_pos, int &discounted_pos, int &discount_power
) {
    change_pos = group_of_thread * stride_cur_group + thread_in_group + stride_prev_group;
    discounted_pos = group_of_thread * stride_cur_group + stride_prev_group - 1;
    discount_power = thread_in_group + 1;
}


template <>
__device__ __forceinline__
void resolve_positions<SUM_DIRECTION_RIGHT>(
    const int &stride_prev_group, const int &stride_cur_group, const int &group_of_thread, const int &thread_in_group,
    int &change_pos, int &discounted_pos, int &discount_power
) {
    change_pos = group_of_thread * stride_cur_group + thread_in_group;
    discounted_pos = group_of_thread * stride_cur_group + stride_prev_group;
    discount_power = stride_prev_group - thread_in_group;
}


template <typename scalar_t, SumDirection sum_direction>
__global__
void discounted_cumsum_kernel_stage(
    torch::PackedTensorAccessor32<scalar_t, 2> x,
    const scalar_t gamma,
    int stage
) {
    const int len = x.size(1);
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_idy >= x.size(0)) {
        return;
    }

    int stride_prev_group = 1 << stage;
    int stride_cur_group = stride_prev_group << 1;

    int group_of_thread = thread_idx >> stage;
    int thread_in_group = thread_idx - (group_of_thread << stage);

    int change_pos, discounted_pos, discount_power;
    resolve_positions<sum_direction>(
        stride_prev_group, stride_cur_group, group_of_thread, thread_in_group,
        change_pos, discounted_pos, discount_power
    );

    if (change_pos >= len || discounted_pos >= len) {
        return;
    }

    x[thread_idy][change_pos] = discounted_sum_power(
        x[thread_idy][change_pos],
        x[thread_idy][discounted_pos],
        gamma,
        discount_power
    );
}


inline
int log2ceil(int x) {
    return (int)ceil(log2((float)x));
}


template <SumDirection sum_direction>
torch::Tensor discounted_cumsum(torch::Tensor x, double gamma) {
    // Minimum required number of threads, assigns them dynamically to respective positions upon each iteration.
    // Results in uncoalesced writes, which is still faster than coalesced writes with half threads idling.

    TORCH_CHECK(x.type().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(0.0 <= gamma && gamma <= 1.0, "Gamma must be in the range [0,1]");

    if (x.size(1) == 0) {
        return x;
    }

    auto y = x.clone();

    const int threads = 64;
    const int nstages = log2ceil(x.size(1));
    const int threads_total_x = 1 << (nstages - 1);
    const dim3 blocks((threads_total_x + threads - 1) / threads, x.size(0));

    for (int stage=0; stage<nstages; stage++) {
        AT_DISPATCH_FLOATING_TYPES(x.type(), "discounted_cumsum_kernel_stage", ([&] {
            discounted_cumsum_kernel_stage<scalar_t, sum_direction><<<blocks, threads>>>(
                y.packed_accessor32<scalar_t, 2>(),
                scalar_t(gamma),
                stage
            );
        }));
    }

    return y;
}


torch::Tensor discounted_cumsum_left(torch::Tensor x, double gamma) {
    return discounted_cumsum<SUM_DIRECTION_LEFT>(x, gamma);
}


torch::Tensor discounted_cumsum_right(torch::Tensor x, double gamma) {
    return discounted_cumsum<SUM_DIRECTION_RIGHT>(x, gamma);
}
