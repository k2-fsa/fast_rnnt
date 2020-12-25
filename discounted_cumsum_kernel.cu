#include <torch/extension.h>


template <typename scalar_t>
__device__ __forceinline__ scalar_t discounted_sum_pow(scalar_t a, scalar_t b, scalar_t gamma, int power) {
    return a + b * pow(gamma, scalar_t(power));
}


__inline__
int log2ceil(int x) {
    return (int)ceil(log2((float)x));
}


template <typename scalar_t>
__global__ void discounted_cumsum_right_kernel_minthreads_stage(
        torch::PackedTensorAccessor32<scalar_t, 2> x,
        const scalar_t gamma,
        int stage
) {
    // Pros: Minimum required number of threads, assigns them dynamically to respective positions upon each iteration.
    // Cons: Uncoalesced writes.

    const int len = x.size(1);
    const int threadidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadidy >= x.size(0)) {
        return;
    }

    int gr_prev_stride = 1 << stage;
    int gr_cur_stride = gr_prev_stride << 1;

    int gr_of_thread = threadidx >> stage;
    int thread_in_gr = threadidx - (gr_of_thread << stage);

    int change_pos = gr_of_thread * gr_cur_stride + thread_in_gr;
    int discounted_pos = gr_of_thread * gr_cur_stride + gr_prev_stride;
    int discount_power = gr_prev_stride - thread_in_gr;

    if (change_pos >= len || discounted_pos >= len) {
        return;
    }

    x[threadidy][change_pos] = discounted_sum_pow(
        x[threadidy][change_pos],
        x[threadidy][discounted_pos],
        gamma,
        discount_power
    );
}


template <typename scalar_t>
__global__ void discounted_cumsum_right_kernel_coalesced_stage(
        torch::PackedTensorAccessor32<scalar_t, 2> x,
        const scalar_t gamma,
        int stage
) {
    // Pros: Coalesced writes.
    // Cons: Threads allocated statically per each element. Half of threads idles upon each iteration.

    const int len = x.size(1);
    const int threadidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadidx >= len || threadidy >= x.size(0)) {
        return;
    }

    int gr_prev_stride = 1 << stage;
    int gr_cur_stride = gr_prev_stride << 1;

    int gr_of_thread = threadidx >> (stage + 1);
    int thread_in_gr = threadidx - (gr_of_thread << (stage + 1));

    int change_pos = threadidx;
    int discounted_pos = gr_of_thread * gr_cur_stride + gr_prev_stride;
    int discount_power = gr_prev_stride - thread_in_gr;

    if (thread_in_gr >= gr_prev_stride || discounted_pos >= len) {
        return;
    }

    x[threadidy][change_pos] = discounted_sum_pow(
        x[threadidy][change_pos],
        x[threadidy][discounted_pos],
        gamma,
        discount_power
    );
}


torch::Tensor discounted_cumsum_right_minthreads(torch::Tensor x, double gamma) {
    // Pros: Minimum required number of threads, assigns them dynamically to respective positions upon each iteration.
    // Cons: Uncoalesced writes.

    TORCH_CHECK(x.type().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(0.0 <= gamma && gamma <= 1.0, "Gamma must be in the range [0,1]");

    if (x.size(1) == 0) {
        return x;
    }

    auto y = x.clone();

    const int threads = 32;
    const int nstages = log2ceil(x.size(1));
    const int threads_total_x = 1 << (nstages - 1);
    const dim3 blocks((threads_total_x + threads - 1) / threads, x.size(0));

    for (int stage=0; stage<nstages; stage++) {
        AT_DISPATCH_FLOATING_TYPES(x.type(), "discounted_cumsum_right_kernel_minthreads_stage", ([&] {
            discounted_cumsum_right_kernel_minthreads_stage<scalar_t><<<blocks, threads>>>(
                y.packed_accessor32<scalar_t, 2>(),
                scalar_t(gamma),
                stage
            );
        }));
    }

    return y;
}


torch::Tensor discounted_cumsum_right_coalesced(torch::Tensor x, double gamma) {
    // Pros: Coalesced writes.
    // Cons: Threads allocated statically per each element. Half of threads idles upon each iteration.

    TORCH_CHECK(x.type().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(0.0 <= gamma && gamma <= 1.0, "Gamma must be in the range [0,1]");

    if (x.size(1) == 0) {
        return x;
    }

    auto y = x.clone();

    const int threads = 32;
    const int nstages = log2ceil(x.size(1));
    const dim3 blocks((x.size(1) + threads - 1) / threads, x.size(0));

    for (int stage=0; stage<nstages; stage++) {
        AT_DISPATCH_FLOATING_TYPES(x.type(), "discounted_cumsum_right_kernel_coalesced_stage", ([&] {
            discounted_cumsum_right_kernel_coalesced_stage<scalar_t><<<blocks, threads>>>(
                y.packed_accessor32<scalar_t, 2>(),
                scalar_t(gamma),
                stage
            );
        }));
    }

    return y;
}
