#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // for getCurrentCUDAStream()
#include <cooperative_groups.h>


#define THREADS_PER_BLOCK 256



/*
  Tiled summing reduction within a warp.  Requires that the thread-block
  be 1-dimensional, i.e.  blockDim.y == blockDim.z == 1.  Does not use
  __syncthreads, so it is safe to call in a subset of threads.
  TODO: we can in principle do this without a buffer, using __shfl_down()
  (see here https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-)
  if CC >= 3.0.

  Args:
      threads_per_tile:  Must be a power of 2 in the interval [1,32].  Summation is
                         within blocks of threads of this size.
       buf:              Pointer to the start of a __shared__ buffer of size
                         blockDim.x, to be used as a temporary within this function.
       val:              The value to be summed
   Return:
       Threads where threadIdx.x % threads_per_tile == 0 will return the sum:
         \sum_{i=0}^{threads_per_tile-1} [val in thread threadIdx.x + i]
       The return value in other threads is undefined.
 */
template <typename scalar_t>
__forceinline__ __device__ scalar_t tiled_warp_reduce_sum(int threads_per_tile,
                                                          __volatile__ scalar_t *buf,
                                                          scalar_t val) {
  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = threads_per_tile / 2; i > 0; i /= 2) {
    buf[threadIdx.x] = val;
    if (threadIdx.x % threads_per_tile < i)
      val += buf[threadIdx.x + i];
  }
  return val; // Only threads with threadIdx.x % threads_per_tile == 0 will
              // return the full sums of their tiles.
}


/*
  Forward of learned_nonlin.  Each thread group handles a single channel (channel
  c = blockIdx.x); the gridDim is (C, nb, 1) where 1 <= nb <= B (nb relates to the
  image within the batch).

  Template args:
      scalar_t: the floating-point type, e.g. float, double, maybe half.

  Args:
      input:  input image, shape (B, C, T) where B is batch size, C is
              the number of channels and T is the time axis.  (For more-than-1d
              convolution setups, T would really be more than 1 axis, reshaped).
      params:  of shape (C, N+1) where N is the number of linear regions in the
               piecewise linear function; params[c][0] is l which is
               a log scale parameter that dictates how far apart
               the discontinuities in the piecewise linear function are,
               and params[c][n+1] for 0 <= n < N are the derivatives
               of the linear parts of the piecewise linear function.
               The discontinuities of the function are at:
                    exp(l) * [ -(N/2 - 1), -(N/2 - 2), ... (N/2 - 1) ]
      output:  The transformed input, shape (B , C, T)
      images_per_thread_block:  The number of images processed by each thread
               block.  The calling code must guarantee that this is a power
               of 2, and that EITHER:
                   THREADS_PER_BLOCK / images_per_thread_block >= T
               OR
                   images_per_thread_block == 1
                .. this is used for a small optimization.

    This kernel is allocated with `extern_buf` containing enough memory
    to store 2*N + 3 values of type scalar_t.

   The blockDim must equal (THREADS_PER_BLOCK, 1, 1)

   The requirements on the grid dimension are:
       gridDim.x == num-channels C (required)
   1 <=  gridDim.y <= B, where B is the number of blocks
       gridDim.z == 1
  When we invoke this kernel, we'll invoke it as:
   learned_nonlin_kernel<<<gridDim, blockDim, bytesShared, stream>>>
   where bytesShared is the number of bytes needed in `extern_buf`:
     bytesShared = sizeof(shared_t) * (2N + 3)
    We also require N + 1 <= THREADS_PER_BLOCK.
 */
extern __shared__ int extern_buf[];

template <typename scalar_t>
__global__
void learned_nonlin_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3> input,  // B, C, T, i.e. batch, channels, time
    torch::PackedTensorAccessor32<scalar_t, 2> params,  // C, N + 1
    torch::PackedTensorAccessor32<scalar_t, 3> output,
    int images_per_thread_block) {  // B, C, T

  const int B = input.size(0),
      C = input.size(1),
      T = input.size(2),
      N = params.size(1) - 1,
      K = N / 2;  // Note: N and K are powers of 2, with K >= 1.

  const int c = blockIdx.x;  // c is channel index

  scalar_t *y_vals = (scalar_t*) extern_buf,  // [N], actually there are 3
                                              // spaces between here and
                                              // `params_buf` for storing scale
                                              // and inv_scale and l == params[c][0].
      *params_buf = (scalar_t*) y_vals + 3 + N;  // [N].  Caution: contains params[c][1] through params[c][N].
                                                 //  params_buf[-1] contains params[c][0] == log of scale;
                                                 // params_buf[-2] and params_buf[-3] contain scale and inv_scale.
  // Load parameters
  if (threadIdx.x <= N)
    params_buf[threadIdx.x - 1] = params[c][threadIdx.x];

  __syncthreads();
  // The easiest way to understand this code is to compare it with the CPU code
  // in learned_nonlin_cpu.cpp.
  // TODO: replace this with easier-to-understand code.
  if ((((int)threadIdx.x & ~(int)64)) == 0) {
    // threadIdx.x == 0 or 64 (we choose 64 because it's >= the max known warp
    // size).  These are in separate warps so we can allow them to do separate
    // jobs.  This code takes linear time in K which is not at all ideal and
    // could be improved if K is largish, but it shouldn't dominate the total
    // time taken if we are processing a lot of data; and anyway, we doubt that
    // K will be need to be more than 4 or 8 or so, so the potential savings are
    // quite small.
    scalar_t scale = exp(params_buf[-1]),
        inv_scale = 1.0 / scale;
    params_buf[-2] = scale;  // both threads write these but it's OK, it's the
                             // same value.
    params_buf[-3] = inv_scale;
    int sign,
        Koffset;  // Koffset == K for threads handling sum_positive and K - 1
                  // for threads handling sum_negative, see
                  // learned_nonlin_cpu.cpp for reference code.  This would be K
                  // + 1 and K respectively, except our params_buf has its index
                  // shifted by one versus params.
    if (threadIdx.x == 0) {  // sum_positive
      sign = 1;
      Koffset = K;
    } else {  // threadIdx.x == 64.  sum_negative.
      scale *= -1;  // this is a local variable..
      sign = -1;
      Koffset = K - 1;
    }
    scalar_t sum = 0.0;
    for (int i = 0; i < K; i++) {
      int isign = i * sign;
      y_vals[K + isign] = sum * scale;
      sum += params_buf[Koffset + isign];
    }
    if (threadIdx.x != 0)  // sum_negative
      y_vals[0] = sum * scale;
  }
  __syncthreads();
  scalar_t inv_scale = params_buf[-3];

  int T_inc = THREADS_PER_BLOCK / images_per_thread_block,
      b_offset = threadIdx.x / T_inc,  // offset within batch
      t_start = threadIdx.x % T_inc;

  for (int b = blockIdx.y * images_per_thread_block + b_offset; b < B;
       b += gridDim.y * images_per_thread_block) {
    // We do "t += THREADS_PER_BLOCK" instead of t += (THREADS_PER_BLOCK /
    // images_per_thread_block) as a small optimization because the only case we
    // really need to loop is when images_per_thread_block == 1:a we only let
    // images_per_thread_block > 1 if T * images_per_thread_block <=
    // THREADS_PER_BLOCK.
    for (int t = t_start; t < T; t += THREADS_PER_BLOCK) {
      scalar_t x = input[b][c][t] * inv_scale + K,
          x_trunc = x;
      if (x_trunc < 0) x_trunc = 0;
      else if (x_trunc >= N) x_trunc = N - 1;
      // C++ rounds toward zero.
      int n = (int) x_trunc;
      // OK, at this point, 0 <= min < N.
      output[b][c][t] = (x - n) * params_buf[n] + y_vals[n];
    }
  }

}



/*
  Summing reduction within a one-dimensional thread block, but with a
  stride of N, so that we separately sum up the values of all threads with
  threadIdx.x % N == 0, with threadIdx.x % N == 1, and so on.  At the end,
  threads with 0 <= threadIdx.x < N contain the sums.

  So this is like tiled summing reduction except that the tiles are
  interspersed with each other.


  Args:
       N:                The number we sum modulo (must be a power of 2 with
                         1 <= N <= blockDim.x), i.e. all threads with
                         threadIdx.x % N == n for some 0 <= n < N have `val` summed.
       buf:              Pointer to the start of a __shared__ buffer of size
                         blockDim.x, to be used as a temporary within this function.
       val:              The value to be summed
  Return:
       Threads where threadIdx.x < N will return the sums (over the threads with
       the same value of threadIdx.x % N);
       the return value in other threads is undefined.
 */
template <typename scalar_t>
__forceinline__ __device__ scalar_t strided_reduce_sum(int N,
                                                       __volatile__ scalar_t *buf,
                                                       scalar_t val) {
  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = blockDim.x / 2; i >= N; i /= 2) {
    buf[threadIdx.x] = val;
    __syncthreads();
    if (threadIdx.x < i)
      val += buf[threadIdx.x + i];
  }
  return val; // Only threads with threadIdx.x < N will return the full sums of
              // their groups.
}

/*
  Backward of learned_nonlin.  Each thread group handles a single channel (channel
  c = blockIdx.x); the gridDim is (C, nb, 1) where 1 <= nb <= B (nb relates to the
  image within the batch).

  Template args:
      scalar_t: the floating-point type, e.g. float, double, maybe half.

  Args:
      input:  input image, shape (B, C, T) where B is batch size, C is
              the number of channels and T is the time axis.  (For more-than-1d
              convolution setups, T would really be more than 1 axis, reshaped).
      params: of shape (C, N+1) where N is the number of linear regions in the
              piecewise linear function; params[c][0] is l which is
              a log scale parameter that dictates how far apart
              the discontinuities in the piecewise linear function are,
              and params[c][n+1] for 0 <= n < N are the derivatives
              of the linear parts of the piecewise linear function.
              The discontinuities of the function are at:
                   exp(l) * [ -(N/2 - 1), -(N/2 - 2), ... (N/2 - 1) ]
      output:  The transformed input, shape (B , C, T)
      images_per_thread_block:  The number of images processed by each thread
               block.  The calling code must guarantee that this is a power
               of 2, and that EITHER:
                   (THREADS_PER_BLOCK / images_per_thread_block >= T  AND
                    THREADS_PER_BLOCK / images_per_thread_block >= N),
               OR
                   images_per_thread_block == 1
                .. this is used for a small optimization.

                ALSO,

    This kernel is allocated with `extern_buf` containing enough memory
    to store 2*N + 3 values of type scalar_t.

   The blockDim must equal (THREADS_PER_BLOCK, 1, 1)

   The requirements on the grid dimension are:
       gridDim.x == num-channels C (required)
   1 <=  gridDim.y <= B, where B is the number of blocks
       gridDim.z == 1
  When we invoke this kernel, we'll invoke it as:
   learned_nonlin_backward_kernel<<<gridDim, blockDim, bytesShared, stream>>>
   where bytesShared is the number of bytes needed in `extern_buf`:
     bytesShared = sizeof(shared_t) * (2N + 3)

   We also require that N <= THREADS_PER_BLOCK (for best performance,
   N should be quite small, like no larger than 8 or so).
   We also require 4 <= N <= 16 for this code!

 */
template <typename scalar_t>
__global__
void learned_nonlin_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3> input,  // B, C, T, i.e. batch, channels, time
    torch::PackedTensorAccessor32<scalar_t, 2> params,  // C, N + 1
    torch::PackedTensorAccessor32<scalar_t, 3> output_grad, // B, C, T
    torch::PackedTensorAccessor32<scalar_t, 3> input_grad, // B, C, T
    // params_grad is of dim (gridDim.y, C, N + 1), we'll sum over dim 0.
    torch::PackedTensorAccessor32<scalar_t, 3> params_grad,
    int images_per_thread_block) {  // B, C, T

  const int B = input.size(0),
      C = input.size(1),
      T = input.size(2),
      N = params.size(1) - 1,
      K = N / 2;  // Note: N and K are powers fo 2, with K >= 1.

  const int c = blockIdx.x; // c is channel index

  scalar_t *y_vals = (scalar_t*) extern_buf,  // [N], actually there are three
                                              // spaces between here and
                                              // `params_buf` for storing scale
                                              // and inv_scale and l == params[c][0].
      *params_buf = (scalar_t*) y_vals + 3 + N;  // [N].  Caution: contains params[c][1] through params[c][N].
                                                 //  params_buf[-1] contains params[c][0] == log of scale;
                                                 // params_buf[-2] and params_buf[-3] contain scale and inv_scale.

  scalar_t x_residual_buf[THREADS_PER_BLOCK];  // x_residual, with 0 <=
                                               // x_residual < 1 for interior
                                               // regions, is the residual part
                                               // of the scaled input, after
                                               // subtracting the integer part.
  scalar_t output_grad_buf[THREADS_PER_BLOCK];
  char n_buf[THREADS_PER_BLOCK];  // for each input in `input_buf`, this stores
                                  // the integer value 0 <= n < N which
                                  // determines which piece of the piecewise
                                  // linear function we are in.

  // this_params_grad and this_y_grad pertain to the 'n' value (i.e. the n'th
  // linear interval) corresponding to n == threadIdx.x % N.  For example, if
  // threadIdx.x == 0, this thread's gradient corresponds to the left-most
  // linear interval.
  scalar_t this_params_grad = 0.0,
      this_y_vals_grad = 0.0;

  // Load parameters
  if (threadIdx.x <= N)
    params_buf[threadIdx.x - 1] = params[c][threadIdx.x];

  __syncthreads();
  // The easiest way to understand this code is to compare it with the CPU code
  // in learned_nonlin_cpu.cpp.

  // This next block computes `y_vals`.
  if ((((int)threadIdx.x & ~(int)32)) == 0) {
    // threadIdx.x == 0 or 32.  These are in separate warps so we can
    // allow them to do separate jobs.  This code takes linear time in K which
    // is not at all ideal and could be improved if K is largish, but it shouldn't
    // dominate the total time taken if we are processing a lot of data;
    // and anyway, we doubt that K will be need to be more than 4 or 8 or so,
    // so the potential savings are quite small.
    scalar_t scale = exp(params_buf[-1]),
        inv_scale = 1.0 / scale;
    params_buf[-2] = scale;  // both threads write these but it's OK, it's the
                             // same value.
    params_buf[-3] = inv_scale;
    int sign,
        Koffset;  // Koffset == K for threads handling sum_positive and K - 1
                  // for threads handling sum_negative, see
                  // learned_nonlin_cpu.cpp for reference code.  This would be K
                  // + 1 and K respectively, except our params_buf has its index
                  // shifted by one versus params.
    if (threadIdx.x == 0) {  // sum_positive
      sign = 1;
      Koffset = K;
    } else {  // threadIdx.x == 32.  sum_negative.
      scale *= -1;  // this is a local variable..
      sign = -1;
      Koffset = K - 1;
    }
    scalar_t sum = 0.0;
    for (int i = 0; i < K; i++) {
      int isign = i * sign;
      y_vals[K + isign] = sum * scale;
      sum += params_buf[Koffset + isign];
    }
    if (threadIdx.x != 0)  // sum_negative
      y_vals[0] = sum * scale;
  }
  __syncthreads();
  scalar_t inv_scale = params_buf[-3];

  int T_inc = THREADS_PER_BLOCK / images_per_thread_block,
      b_offset = threadIdx.x / T_inc;  // offset within batch

  for (int b = blockIdx.y * images_per_thread_block + b_offset; b < B;
       b += gridDim.y * images_per_thread_block) {

    // The following will loop just once if images_per_thread_block > 1.  If
    // images_per_thread_block == 1 and T > THREADS_PER_BLOCK, we will loop
    // multiple times.  We want to keep all threads active so that output_grad
    // will be set to zero for excess threads, and thus won't contribute to
    // this_params_grad or this_y_vals_grad.
    for (int t_offset = 0; t_offset < T; t_offset += THREADS_PER_BLOCK) {
      int t = threadIdx.x % T_inc + t_offset;
      scalar_t this_output_grad = 0.0, x = 0.0;
      if (t < T)
        this_output_grad = output_grad[b][c][t];

      // The reason we use t % T here rather than only invoking this in some
      // threads, is so that the un-needed threads will have a similar
      // distribution over 'n' to the needed threads, which will hopefully avoid
      // excessive work for some particular 'n' value if too many x values had
      // the same 'n'.  It might be better to set n to an invalid value for
      // out-of-range threads, but as it is, if we are to properly handle
      // N==16 we don't have enough bits available in `src_indexes` to do this.
      x = input[b][c][t % T] * inv_scale + K;

      output_grad_buf[threadIdx.x] = this_output_grad;

      scalar_t x_trunc = x;
      if (x_trunc < 0) x_trunc = 0;
      else if (x_trunc >= N) x_trunc = N - 1;
      // C++ rounds toward zero.
      int n = (int)x_trunc;
      n_buf[threadIdx.x] = (char)n;

      scalar_t x_residual = x - n;
      x_residual_buf[threadIdx.x] = x_residual;

      // OK, at this point, 0 <= min < N.
      // The forward code did:
      // output[b][c][t] = (x - n) * params_buf[n] + y_vals[n];

      if (t < T)
        input_grad[b][c][t] = this_output_grad * params_buf[n];

      int this_block_start = threadIdx.x & ~(N-1),  // ==  N * (threadIdx.x / N),
          this_n = threadIdx.x & (N-1); // == threadIdx.x % N.
      // this_n is the n value that this thread accumulates gradients for;
      // it is responsible for output_grads in the block of threads
      // from this_block_start to this_block_start+N-1.


      // SYNC POINT At this point there is an implicit within-warp
      // synchronization (Note: implicit warp synchronization is considered not
      // future-proof).  Threads above have written to n_buf, and threads below
      // will read from it; but we don't need to explicitly synchronize for now
      // because the reads/writes are among threads in a group of N threads with
      // (4 <= N <= 16); and 16 is less than the warp size which is 32 or 64.

      // src_indexes will contain up to 16 16-bit numbers, stored starting in its
      // least significant bits.  It will store all the offsets within this
      // block of N, where the 'n' value equals this_n.
      uint64_t src_indexes = 0;
      // num_src is the number of numbers in `src_indexes`.  We need to store a
      // separate counter because zero is a valid index and if we are to support
      // N == 16 we don't have bits to spare in src_indexes to store some kind
      // of marker.
      int num_src = 0;

      // This loop always does N statements, but they should be relatively fast
      // ones since the computation per n value is minimal and there is little
      // I/O.  We are figuring out the subset of our block of N elements,
      // which this particular thread value is responsible for (because they
      // have n == this_n), and storing them in `src_indexes` and `num_src`.
      for (int i = 0; i < N; i += 4) {
        uint32_t n_block_of_4 = *reinterpret_cast<uint32_t*>(n_buf + this_block_start + i);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          // CUDA is little endian
          char n = (char)(n_block_of_4 >> (8*j));
          if (n == this_n) {
            // We require that N <= 16, so 4 bits is enough to store src_idx.
            src_indexes = (src_indexes << 4) | (i + j);
            ++num_src;
          }
          // Note: if, for out-of-range threads, we had values not in [0..N-1] in
          // n_buf they won't end up mattering even though they are read here,
          // because they won't equal this_n.  For values 0 <= n < N originating
          // in out-of-range threads, the value won't matter because the
          // corresponding value in output_grad_buf will be zero.
        }
      }

      // While num_src could theoretically be as large as N, the hope is that no
      // thread in any given warp actually loops that many times.  Once all
      // threads in the warp are finished looping, we can continue.  It is OK
      // for different warps to get out of sync here; we could be looping over a
      // number of images, and the hope is that different warps will reach the
      // end of the outer loop at around the same time because their variations
      // in speed will average out.
      for (; num_src > 0; --num_src, src_indexes >>= 4) {
        int src_idx = src_indexes & 0xF,
            src_thread = this_block_start + src_idx;
        scalar_t output_grad = output_grad_buf[src_thread],
            x_residual = x_residual_buf[src_thread];
        // Backprop for: output = x_residual * params_buf[n] + y_vals[n].
        // Here, n == this_n; this is how we selected these `src_idx` values.
        this_params_grad += output_grad * x_residual;
        this_y_vals_grad += output_grad;
      }
    }
  }

  __syncthreads();  // sync threads because we are about to re-use
                    // output_grad_buf for reduction.

  this_params_grad = strided_reduce_sum(N, output_grad_buf, this_params_grad);
  this_y_vals_grad = strided_reduce_sum(N, output_grad_buf, this_y_vals_grad);

  __syncthreads();  // sync threads because we are about to re-use
                    // output_grad_buf.

  // Re-use some buffers..
  scalar_t *params_grad_buf = x_residual_buf,  // [N]
      *y_vals_grad_buf = output_grad_buf;   // [N]

  if (threadIdx.x < N) {
    // There is an offset of 1 between the 'n' values and
    // the position in 'params'.  To keep the backprop code similar to the CPU
    // backprop code we restore that offset here, i.e. use the same layout
    // as the params.
    params_grad_buf[threadIdx.x + 1] = this_params_grad;
    y_vals_grad_buf[threadIdx.x] = this_y_vals_grad;
  }


  // This next block does backprop relating to `y_vals`.  Comparing with the CPU
  // version (call this the "reference code") is the best way to understand this (this code is just a
  // modification of that).
  {
    // Thread 0 is responsible for parts of the reference code that involve "sum_positive_grad";
    // thread 64 is responsible for parts of the reference code that involve "sum_negative_grad";
    scalar_t scale_grad = 0.0,
        scale = params_buf[-2];

    if (threadIdx.x == 0) {
      scalar_t sum_positive_grad = 0.0;
      for (int i = K - 1; i >= 0; i--) {
        // This is like the CPU code but with an offset of 1 for 'params_buf'
        // versus 'params_a'.
        params_grad_buf[1 + K + i] += sum_positive_grad * scale;
        scale_grad += sum_positive_grad * params_buf[K + i];
        sum_positive_grad += y_vals_grad_buf[K + i];
      }
      params_grad_buf[0] += scale * scale_grad;
    } else if (threadIdx.x == 64) {
      scalar_t sum_negative_grad = y_vals_grad_buf[0];
      for (int i = K - 1; i >= 0; i--) {
        // This is like the CPU code but with an offset of 1 for 'params_buf'
        // versus 'params_a'.
        params_grad_buf[K - i] -= sum_negative_grad * scale;
        scale_grad -= sum_negative_grad * params_buf[K - 1 - i];
        sum_negative_grad += y_vals_grad_buf[K - i];
      }
    }
    __syncthreads();
    if (threadIdx.x == 64)
      params_grad_buf[0] += scale * scale_grad;
    __syncthreads();
  }

  if (threadIdx.x <= N) {
    params_grad[blockIdx.y][c][threadIdx.x] = params_grad_buf[threadIdx.x];
  }
}




torch::Tensor learned_nonlin_cuda(torch::Tensor input,
                                  torch::Tensor params) {

  TORCH_CHECK(input.dim() == 3, "input must be 3-dimensional");
  TORCH_CHECK(params.dim() == 2, "params must be 2-dimensional.");
  TORCH_CHECK(params.size(1) >= 3 &&
              ((params.size(1) - 1) & (params.size(1) - 2)) == 0,
              "params.size(1) has invalid value, must be a power of 2 plus 1.");
  TORCH_CHECK(params.size(0) == input.size(1),
              "params vs input channels mismatch");

  TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(params.device().is_cuda(), "Params must be a CUDA tensor");


  const int B = input.size(0),
      C = input.size(1),
      T = input.size(2),
      N = params.size(1) - 1;

  auto scalar_t = input.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(input.device());

  torch::Tensor output = torch::empty({B, C, T}, opts);

  if (C * B * T == 0)
    return output;

  int images_per_thread_block = 1;
  while (images_per_thread_block * 2 * T <= THREADS_PER_BLOCK)
    images_per_thread_block *= 2;

  int grid_dim_y = 1;
  // If the number of channels is quite small (<128) we can launch more thread
  // groups, splitting on the batch index.
  while (C * grid_dim_y < 128)
    grid_dim_y *= 2;

  // B_reduced is the max number of thread-groups per channel that would have
  // any work to do.  If grid_dim_y is more than this, we reduce it to avoid
  // launching kernels with nothing to do.
  int B_reduced = (B + images_per_thread_block - 1) / images_per_thread_block;
  if (grid_dim_y > B_reduced)
    grid_dim_y = B_reduced;

  int shared_mem_numel = 2 * N + 3;

  if (false)
    std::cout << "C,B,T,N = " << C << "," << B << "," << T << "," << N
              << ", images_per_thread_block = " << images_per_thread_block
              << ", grid_dim_y = " << grid_dim_y
              << "\n";

  TORCH_CHECK(THREADS_PER_BLOCK / images_per_thread_block >= T ||
              images_per_thread_block == 1,
              "Code error");

  TORCH_CHECK(N + 1 <= THREADS_PER_BLOCK,
              "Values of N this large are not supported.");

  dim3 gridDim(C, grid_dim_y, 1);

  // blockDim is scalar, just THREADS_PER_BLOCK.
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "learned_nonlin_kernel", ([&] {
        learned_nonlin_kernel<scalar_t><<<gridDim, THREADS_PER_BLOCK, sizeof(scalar_t) * shared_mem_numel, at::cuda::getCurrentCUDAStream()>>>(
              input.packed_accessor32<scalar_t, 3>(),
              params.packed_accessor32<scalar_t, 2>(),
              output.packed_accessor32<scalar_t, 3>(),
              images_per_thread_block);
      }));
  return output;
}



std::vector<torch::Tensor> learned_nonlin_backward_cuda(torch::Tensor input,
                                                        torch::Tensor params,
                                                        torch::Tensor output_grad) {
  TORCH_CHECK(input.dim() == 3, "input must be 3-dimensional");
  TORCH_CHECK(params.dim() == 2, "params must be 2-dimensional.");
  TORCH_CHECK(params.size(1) >= 3 &&
              ((params.size(1) - 1) & (params.size(1) - 2)) == 0,
              "params.size(1) has invalid value, must be a power of 2 plus 1.");
  TORCH_CHECK(params.size(0) == input.size(1),
              "params vs input channels mismatch");
  TORCH_CHECK(output_grad.dim() == 3 && output_grad.size(0) == input.size(0) &&
              output_grad.size(1) == input.size(1) &&
              output_grad.size(2) == input.size(2),
              "output_grad and input have mismatched dim.");

  TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(output_grad.device().is_cuda(), "output_grad must be a CUDA tensor");
  TORCH_CHECK(params.device().is_cuda(), "Params must be a CUDA tensor");


  const int B = input.size(0),
      C = input.size(1),
      T = input.size(2),
      N = params.size(1) - 1;

  TORCH_CHECK(N >= 4, "This backward code requires N >= 4");
  TORCH_CHECK(N <= 16, "This backward code currently requires N <= 16");

  auto scalar_t = input.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(input.device());


  torch::Tensor input_grad = torch::empty({B, C, T}, opts);

  if (C * B * T == 0) {
    return std::vector<torch::Tensor>({input_grad,
            torch::empty({C, N + 1})});
  }

  int images_per_thread_block = 1;
  while (images_per_thread_block * 2 * T <= THREADS_PER_BLOCK &&
         images_per_thread_block * 2 * N <= THREADS_PER_BLOCK)
    images_per_thread_block *= 2;

  int grid_dim_y = 1;
  // If the number of channels is quite small (<128) we can launch more thread
  // groups, splitting on the batch index.
  while (C * grid_dim_y < 128)
    grid_dim_y *= 2;

  // B_reduced is the max number of thread-groups per channel that would have
  // any work to do.  If grid_dim_y is more than this, we reduce it to avoid
  // launching kernels with nothing to do.
  int B_reduced = (B + images_per_thread_block - 1) / images_per_thread_block;
  if (grid_dim_y > B_reduced)
    grid_dim_y = B_reduced;

  int shared_mem_numel = 2 * N + 3;

  if (false)
    std::cout << "C,B,T,N = " << C << "," << B << "," << T << "," << N
              << ", images_per_thread_block = " << images_per_thread_block
              << ", grid_dim_y = " << grid_dim_y
              << "\n";

  TORCH_CHECK(THREADS_PER_BLOCK / images_per_thread_block >= T ||
              images_per_thread_block == 1,
              "Code error");


  torch::Tensor params_grad = torch::empty({grid_dim_y, C, N + 1}, opts);

  dim3 gridDim(C, grid_dim_y, 1);

  // blockDim is scalar, just THREADS_PER_BLOCK.
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "learned_nonlin_backward_kernel", ([&] {
        learned_nonlin_backward_kernel<scalar_t><<<gridDim, THREADS_PER_BLOCK, sizeof(scalar_t) * shared_mem_numel, at::cuda::getCurrentCUDAStream()>>>(
            input.packed_accessor32<scalar_t, 3>(),
            params.packed_accessor32<scalar_t, 2>(),
            output_grad.packed_accessor32<scalar_t, 3>(),
            input_grad.packed_accessor32<scalar_t, 3>(),
            params_grad.packed_accessor32<scalar_t, 3>(),
            images_per_thread_block);
      }));

  params_grad = at::sum(params_grad, {0});
  return std::vector<torch::Tensor>({input_grad, params_grad});
}
