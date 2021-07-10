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
       Threads where blockDim.x % threads_per_tile == 0 will return the sum:
         \sum_{i=0}^{threads_per_tile-1} [val in thread threadIdx.x + i]
       Return value in other threads is undefined.
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
  Forward of learned_nonlin.  Each thread group handles a single channel (equal
  to blockIdx.x); the gridDim is (C, nb) where 1 <= nb <= B (nb relates to the batch).

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
    to store 2*N values of type scalar_t.

   The blockDim must equal (THREADS_PER_BLOCK, 1, 1)

   The requirements on the grid dimension are:
       gridDim.x == num-channels C (required)
   1 <=  gridDim.y <= B, where B is the number of blocks
       gridDim.z == 1
  When we invoke this kernel, we'll invoke it as:
   learned_nonlin_forward<<<gridDim, blockDim, bytesShared, stream>>>
   where bytesShared is the number of bytes needed in `extern_buf`:
     bytesShared = sizeof(shared_t) * (2N + 3)
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
      K = N / 2;  // Note: N and K are powers fo 2, with K >= 1.

  const int c = blockIdx.x; // c is channel index

  scalar_t *y_vals = (scalar_t*) extern_buf,  // [N], actually there are two
                                              // spaces between here and
                                              // `params_buf` for storing scale
                                              // and inv_scale.
      *params_buf = (scalar_t*) y_vals + 3 + N;  // [N].  Caution: contains params[c][1] through params[c][N].
                                                 //  params_buf[-1] contains params[c][0] == log of scale;
                                                 // params_buf[-2] and params_buf[-3] contain scale and inv_scale.
  // Load parameters
  for (int n = threadIdx.x; n <= N; n += THREADS_PER_BLOCK) {
    params_buf[n - 1] = params[c][n];
  }
  __syncthreads();
  // The easiest way to understand this code is to compare it with the CPU code
  // in learned_nonlin_cpu.cpp.
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
    y_vals[0] = y_vals[1];  // Both threads do this but it's OK.
  }
  __syncthreads();
  scalar_t inv_scale = params_buf[-3];

  int T_inc = THREADS_PER_BLOCK / images_per_thread_block,
      image_offset = threadIdx.x / T_inc,
      t_start = threadIdx.x % T_inc;

  for (int b = blockIdx.y * images_per_thread_block + image_offset;
       b < B; b += gridDim.y * images_per_thread_block) {
    // We do "t += THREADS_PER_BLOCK" instead of t += (THREADS_PER_BLOCK /
    // images_per_thread_block) as a small optimization because the only case we
    // really need to loop is when images_per_thread_block == 1:a we only let
    // images_per_thread_block > 1 if T * images_per_thread_block <=
    // THREADS_PER_BLOCK.
    for (int t = t_start; t < T; t += THREADS_PER_BLOCK) {
      scalar_t x = input[b][c][t] * inv_scale + K;
      int min = 0, diff = K;
      while (diff > 0) {
        int mid = min + diff;
        if (x >= mid)
          min = mid;
        diff = diff >> 1;
      }
      // OK, at this point, 0 <= min < 2*K.
      scalar_t y = (x - (scalar_t)min) * params_buf[min] + y_vals[min];
      output[b][c][t] = y;
    }
  }

}


/*
  Backward of learned_nonlin.  Each thread group handles a single channel (equal
  to blockIdx.x), and loops over patches of the output and over the image n
  within the batch (different thread groups may be responsible for different
  subsets of patches and/or images, see docs of gridDim below).

  If you want to understand this code, you should first understand the forward
  code.  Here are some points to understand how this works:

  First, understand the difference between the patch of size patchH by
  patchW, which is the basic patch size that is related to the blockDim.x,
  and the padded patch size ppatchH and ppatchW, where:
     ppatchH = patchH + kH - 1
     ppatchW = patchW + kW - 1.

  In the forward pass, we dealt with a patch of output and a padded patch of
  input.  In this backward-pass code, when computing the `grad_input` we deal
  with a patch of input and a padded patch of output (this ensures that
  different thread-blocks write to distinct patches of `grad_input`).  But this
  approach is not sufficient to update `grad_pos_add` and `grad_pos_mul`,
  because it's possible for elements of the zero-padding of `input` to
  contribute to `grad_pos_add` and `grad_pos_mul`.  So when computing the
  gradients for those quantities, we actually use a padded input patch and an
  un-padded output patch.  This requires that we load into shared memory the
  padded versions of both input and grad_output.


  Template args:
      scalar_t: the floating-point type, e.g. float, double, maybe half.

  Args:
      input [in]:  input image, shape (N, 2*C, H, W)
      pos_add [in]:  positional encoding, additive part,  shape (C, kH, kW)
      pos_mul [in]:  positional encoding, multiplicative part, shape (C, kH, kW)
      grad_output [in]: the gradient w.r.t. the output of the forward pass, shape (N, C, H, W)
      grad_input [out]: the gradient w.r.t. the input, of shape N, 2*C, H, W
      grad_pos_add [out]: the gradient w.r.t. pos_add, indexed [block][c][kh][kw],
            of shape num_blocks, C, kH, kW,
            where `block` is an index we'll later sum over, that corresponds to
            the identity of the thread-block (except, not including the channel
            dimension == gridDim.x).  So, block == blockIdx.z * gridDim.y + blockIdx.y,
            and num_blocks == gridDim.y * gridDim.z.
      grad_pos_mul [out]: the gradient w.r.t. pos_mul, like grad_pos_add above.
      patchH: the height of the patch size this kernel operates on (prior to padding)
      patchW: the width of the patch size this kernel operates on (prior to padding)
      threads_per_pixel: the number of threads assigned to compute each pixel
              of grad_input.  Require patchH * patchW * threads_per_pixel <= blockDim.x
              and threads_per_pixel must be a power of 2 in the interval [1,32].
      threads_per_kernel_pos: the number of threads assigned to compute each kernel
              position of grad_pos_add and grad_pos_mul.
              Require kH * kW * threads_per_kernel_pos <= blockDim.x,
              and threads_per_kernel_pos must be a power of 2 in the interval [1,32].
              This requires that kH * kW must not be greater than 1024.

  Note: kH and kW must both be odd so that it's clear how to zero-pad.

  The thread-block should have one dimension (x); see docs for threads_per_pixel
  and threads_per_kernel_pos for requirements on blockDim.x.  Also, blockDim.x
  must be an exact multiple of 64, so we can divide the threads by 2 and they
  will be in different warps.

  The requirements on the grid dimension are:
       gridDim.x == num-channels C (required)
       gridDim.y <= num-patches per image (recommended)
       gridDim.z <= batch-size N (recommended)
  When we invoke this kernel, we'll invoke it as:
   learned_nonlin_forward<<<gridDim, blockDim, bytesShared, stream>>>
  where bytesShared is the number of bytes needed in `extern_buf`:

   bytesShared = sizeof(shared_t) * numel, where
    numel = 4 * (kH * kW) + 3 * (ppatchH * ppatchW) + blockDim.x
 */
template <typename scalar_t>
__global__
void learned_nonlin_kernel_backward(
    torch::PackedTensorAccessor32<scalar_t, 4> input,  // N, 2*C, H, W
    torch::PackedTensorAccessor32<scalar_t, 3> pos_add,  // C, kH, kW
    torch::PackedTensorAccessor32<scalar_t, 3> pos_mul,  // C, kH, kW
    torch::PackedTensorAccessor32<scalar_t, 4> grad_output,  // N, C, H, W
    torch::PackedTensorAccessor32<scalar_t, 4> grad_input,  // N, 2*C, H, W
    torch::PackedTensorAccessor32<scalar_t, 4> grad_pos_add, // block, C, kH, kW, see above for `block`
    torch::PackedTensorAccessor32<scalar_t, 4> grad_pos_mul, // block, C, kH, kW, see above for `block`
    int patchH,  // non-padded patch height
    int patchW,  // non-padded patch width
    int threads_per_pixel,
    int threads_per_kernel_pos) {

  const int H = input.size(2),
      W = input.size(3),
      kH = pos_add.size(1),
      kW = pos_add.size(2),
      npatchH = (H + patchH - 1) / patchH,  // num patches in vertical dim
      npatchW = (W + patchW - 1) / patchW,  // num patches in horizontal dim
      npatch = npatchH * npatchW;  // total number of patches per image

  // Channel index.
  const int c = blockIdx.x;
  // We don't need to check the range of `c` because we set gridDim.x to the
  // exact number of channels.

  const int ppatchH = patchH + kH - 1,  // ppatchH is the padded patch height.
      ppatchW = patchW + kW - 1,  // ppatchW is the padded patch width
      patch_size = patchH * patchW,  // un-padded patch size
      ppatch_size = ppatchH * ppatchW;  // padded patch size

  // `extern_buf` is general-purpose shared memory, which we'll divide between
  // various buffers.

  // these are pointers to __shared__ memory; the compiler should
  // be able to figure this out.
  scalar_t
      *pos_add_buf = (scalar_t*)extern_buf,     // pos_add positional-encoding / kernel parameters,
                                // indexed [kh*kW + kw] where kh and kw are vertical
                                // and horizontal positions in the kernel.
      *pos_mul_buf = pos_add_buf + (kH * kW), // pos_mul positional-encoding / kernel parameters,
                                              // indexed [kh*kW + kw] where kh and kw are vertical
                                              // and horizontal positions in the kernel.
      *src_img_buf = pos_mul_buf + (kH * kW),    // version of input image that relates to source position,
                             // of size [ppatch_size], indexed [h*ppatchW + w],
                             // where the 'h' and 'w' indexes are into the zero-padded input
                             // image.
      *dest_img_buf = src_img_buf + ppatch_size,  // version of input image that relates to destinatioon position
      *grad_output_buf = dest_img_buf + ppatch_size, // output gradient for padded patch, indexed [h*ppatchW + w]
      *grad_pos_add_buf = grad_output_buf + ppatch_size,  // total grad for pos_add for this thread block, indexed [kh*kW + kw]
      *grad_pos_mul_buf = grad_pos_add_buf + (kH * kW),  // total grad for pos_mul for this thread block, indexed [kh*kW + kw]
      *reduce_buf = grad_pos_mul_buf + (kH * kW);  // buffer for reduction over threads, size == blockDim.x


  // pos_in_patch will be interpreted as h_in_patch * patchW + w_in_patch.
  int pos_in_patch = threadIdx.x / threads_per_pixel;

  // Load parts of the kernel parameters pos_add and pos_mul into shared memory,
  // in pos_add_buf and pos_mul_buf; zero the corresponding gradient buffers.
  // We know that blockDim.x >= kH * kW, see threads_per_kernel_pos.

  for (int i = threadIdx.x % (blockDim.x / 2); i < kH * kW; i += (blockDim.x / 2)) {
    int kh = i / kW, kw = i % kW;
    if (threadIdx.x < blockDim.x / 2) {  // First half of threads take care of pos_add..
      pos_add_buf[i] = pos_add[c][kh][kw];
      grad_pos_add_buf[i] = 0.0;
    } else {  // Second half take care of pos_mul... there is no warp divergence
              // because we make sure blockDim.x is a multiple of 64.
      pos_mul_buf[i] = pos_mul[c][kh][kw];
      grad_pos_mul_buf[i] = 0.0;
    }
  }

  // n is the index within the batch of images.  Loop to make sure we cover all
  // images in the batch.  input.size(0) is the batch size N.  All threads in
  // the thread-block loop the same number of times.
  for (int n = blockIdx.z; n < input.size(0); n += gridDim.z) {

    // Loop over the patch within the output image.  All threads in the
    // thread-block loop the same number of times.
    for (int patch_idx = blockIdx.y; patch_idx < npatch; patch_idx += gridDim.y) {
      // (patch_h_offset, patch_w_offset) are the (vertical, horizontal) indexes
      // of the lowest-numbered pixel in the *un-padded* patch that this thread
      // block is responsible for.  (We'll actualy be loading the padded patches
      // into memory, so be careful).
      int patch_h_offset = (patch_idx / npatchW) * patchH,
          patch_w_offset = (patch_idx % npatchW) * patchW;

      // This __syncthreads() is only necessary if we have already looped at
      // least once over n or patch_idx: it's in case other threads are still
      // using the `src_img_buf` or `dst_img_buf` buffers for a previous patch.
      __syncthreads();

      // Load the 'src' and 'dest' versions of the padded patch into
      // shared-memory buffers, and also the output gradient.
      for (int i = threadIdx.x % (blockDim.x / 2);
           i < ppatch_size; i += (blockDim.x / 2)) {
        int h_in_ppatch = i / ppatchW,
            w_in_ppatch = i % ppatchW;
        int h = patch_h_offset + h_in_ppatch - (kH / 2),  // kH / 2 is offset due to padding
            w = patch_w_offset + w_in_ppatch - (kW / 2);

        if (threadIdx.x < blockDim.x / 2) {  // The first half of the threads of the block
                                             // load `input`
          scalar_t src_val = scalar_t(0),
              dest_val = scalar_t(0);
          if ((unsigned int)h < (unsigned int)H &&  // h >= 0 && h < H
              (unsigned int)w < (unsigned int)W) {  // w >= 0 && w < W
            int C = grad_output.size(1);
            src_val = input[n][c][h][w];
            dest_val = input[n][c + C][h][w];
          }
          src_img_buf[i] = src_val;
          dest_img_buf[i] = dest_val;
        } else {  // second half of threads load `grad_output`.  We require
                  // blockDim.x be an even multiple of the warp size, so there
                  // is no warp divergence here.
          scalar_t grad_output_val = scalar_t(0);
          if ((unsigned int)h < (unsigned int)H &&
              (unsigned int)w < (unsigned int)W)
            grad_output_val = grad_output[n][c][h][w];
          grad_output_buf[i] = grad_output_val;
        }
      }
      // make sure all threads haave written to `src_img_buf`, `dest_img_buf` and
      // `grad_output_buf`.
      __syncthreads();

      scalar_t grad_input_src_sum = 0.0,  // grad for channel c, for our pixel
                                          // of `input` (contribution of this
                                          // thread)
          grad_input_dest_sum = 0.0;   // grad for channel c + C, for our pixel
                                       // of `input` (contribution of this thread)
      if (pos_in_patch < patch_size) {
        // This block computes `grad_input_src_sum` and `grad_input_dest_sum`
        // The num-threads for the backward kernel may not be an exact multiple
        // of patch_size, wo we need the if-guard.

        int h_in_patch = pos_in_patch / patchW,
            w_in_patch = pos_in_patch % patchW,
            h_in_ppatch = h_in_patch + kH / 2,
            w_in_ppatch = w_in_patch + kW / 2,
            pos_in_ppatch = h_in_ppatch * ppatchW + w_in_ppatch;

        // this_dest_val is the `destination` version of our current pixel; this
        // is an input.  It gets added to each src pixel, prior to the relu, in
        // the loop below.
        // this_src_val is the `src` version of our current pixel; it contributes
        // to the outputs of other pixels.
        scalar_t this_dest_val = dest_img_buf[pos_in_ppatch],
            this_src_val = src_img_buf[pos_in_ppatch];

        for (int pos_in_kernel = threadIdx.x % threads_per_pixel;
             pos_in_kernel < (kH * kW);
             pos_in_kernel += threads_per_pixel) {

          int h_in_kernel = pos_in_kernel / kW,
              w_in_kernel = pos_in_kernel % kW;

          // This is actually more like cross-correlation, as we don't have a
          // negative sign on the h and w indexes in the kernel.
          int src_h_in_ppatch = h_in_patch + h_in_kernel,
              src_w_in_ppatch = w_in_patch + w_in_kernel;
          int src_pos_in_ppatch = src_h_in_ppatch * ppatchW + src_w_in_ppatch;

          scalar_t src_val = src_img_buf[src_pos_in_ppatch],
              pos_add_val = pos_add_buf[pos_in_kernel],
              pos_mul_val = pos_mul_buf[pos_in_kernel];
          scalar_t relu = (src_val + this_dest_val + pos_add_val);
          if (relu >= 0.0) {
            scalar_t this_grad_output = grad_output_buf[pos_in_ppatch];
            grad_input_dest_sum += this_grad_output * pos_mul_val;
          }
          // To compute a contribution to "this_input_src_grad", we need to
          // consider the contribution to the destination pixel that it would
          // have contributed to with this same offset.
          // We have to flip the offsets: instead of "+ h_in_kernel",
          // we use (kH - 1) - h_in_kernel,.
          int dest_h_in_ppatch = h_in_patch + (kH - 1) - h_in_kernel,
              dest_w_in_ppatch = w_in_patch + (kW - 1) - w_in_kernel,
              dest_pos_in_ppatch = dest_h_in_ppatch * ppatchW + dest_w_in_ppatch;
          scalar_t dest_val = dest_img_buf[dest_pos_in_ppatch];
          relu = dest_val + this_src_val + pos_add_val;
          if (relu >= 0.0) {
            scalar_t dest_grad_output = grad_output_buf[dest_pos_in_ppatch];
            grad_input_src_sum += dest_grad_output * pos_mul_val;
          }
        }
      }
      // Aggregate `grad_input_src_sum` over threads, if needed; and write the
      // result to `grad_input`.
      // h and w are un-padded indexes into the entire image.
      int h = patch_h_offset + pos_in_patch / patchW,
          w = patch_w_offset + pos_in_patch % patchW;

      if (h < H && w < W) {
        grad_input_src_sum = tiled_warp_reduce_sum(threads_per_pixel,
                                                   reduce_buf,
                                                   grad_input_src_sum);
        grad_input_dest_sum = tiled_warp_reduce_sum(threads_per_pixel,
                                                    reduce_buf,
                                                    grad_input_dest_sum);
        if (threadIdx.x % threads_per_pixel == 0) {
          grad_input[n][c][h][w] = grad_input_src_sum;
          int C = grad_output.size(1);
          grad_input[n][c + C][h][w] = grad_input_dest_sum;
        }
      }

      // OK, we are done computing grad_input for this patch.  Now
      // we need to contribute the contributions to grad_pos_add_buf
      // and grad_pos_mul_buf for this patch.
      // 0 <= pos_in_kernel < (kH * kW).
      int pos_in_kernel = threadIdx.x / threads_per_kernel_pos;
      scalar_t this_grad_pos_add = 0.0,
              this_grad_pos_mul = 0.0;
      if (pos_in_kernel < (kH * kW)) {
        int kh = pos_in_kernel / kW,
            kw = pos_in_kernel % kW;

        // This group of (threads_per_kernel_pos) threads is responsible
        // for position (kh, kw) in the kernel; we iterate over the patch
        // (an un-padded patch of output).
        scalar_t pos_add_val = pos_add_buf[pos_in_kernel],
                 pos_mul_val = pos_mul_buf[pos_in_kernel];

        for (int pos_in_patch = threadIdx.x % threads_per_kernel_pos;
             pos_in_patch < patch_size; pos_in_patch += threads_per_kernel_pos) {
          // We are working out the contribution to the gradients for pos_add
          // and pos_mul; we let `pos_in_patch` correspond to the *output*
          // position, and work out the input position based on gthe kernel position.

          int h_in_patch = pos_in_patch / patchW,
              w_in_patch = pos_in_patch % patchW;

          // pos_in_ppatch is the position in the padded patch corresponding to
          // `pos_in_patch`.
          int pos_in_ppatch = (h_in_patch + kH / 2) * ppatchW + (w_in_patch + kW / 2);
          scalar_t dest_val = dest_img_buf[pos_in_ppatch];
          int src_pos_in_ppatch = (h_in_patch + kh) * ppatchW + (w_in_patch + kw);
          scalar_t src_val = src_img_buf[src_pos_in_ppatch];

          scalar_t relu = dest_val + src_val + pos_add_val;
          if (relu >= 0.0) {
            scalar_t this_grad_output = grad_output_buf[pos_in_ppatch];
            this_grad_pos_add += this_grad_output * pos_mul_val;
            this_grad_pos_mul += this_grad_output * relu;
          }
        }
        this_grad_pos_add = tiled_warp_reduce_sum(
            threads_per_kernel_pos, reduce_buf, this_grad_pos_add);
        this_grad_pos_mul = tiled_warp_reduce_sum(
            threads_per_kernel_pos, reduce_buf, this_grad_pos_mul);
        if (threadIdx.x % threads_per_kernel_pos == 0) {
          grad_pos_add_buf[pos_in_kernel] += this_grad_pos_add;
          grad_pos_mul_buf[pos_in_kernel] += this_grad_pos_mul;
        }
      }
    }
  }

  __syncthreads(); // make sure all threads have written to grad_pos_add_buf and
                   // grad_pos_mul_buf.
  int block = blockIdx.z * gridDim.y + blockIdx.y;

  int kernel_pos = threadIdx.x;
  if (kernel_pos < (kH * kW)) {
    int kh = kernel_pos / kW,
        kw = kernel_pos % kW;
    grad_pos_add[block][c][kh][kw] = grad_pos_add_buf[kernel_pos];
    grad_pos_mul[block][c][kh][kw] = grad_pos_mul_buf[kernel_pos];
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

  // TODO: make this empty

  torch::Tensor output = torch::ones({B, C, T}, opts);

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
                                                        torch::Tensor grad_output) {

  /*
  TORCH_CHECK(input.dim() == 4, "input must be 4-dimensional");
  TORCH_CHECK(pos_add.dim() == 3, "pos_add must be 3-dimensional.");
  TORCH_CHECK(pos_mul.dim() == 3, "pos_add must be 3-dimensional.");
  TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
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


  // Work out the configuration to call the kernel with..
  int patchH = std::min(H, kH),  // output patch height
      patchW = std::min(W, kW);  // output patch width
  // We don't want the height or width of the patch to be less than the kernel
  // width, or the padding will make the input-patch size more than twice the
  // output-patch size.
  // We aim for the output-patch size to be more than 128; this is not something
  // very exact, but it roughly corresponds to us wanting to have up to 4 threads
  // per output pixel, and the limitation of 512 threads per thread-block which
  // we impose so that we can run on architectures with little shared memory.
  while (patchW < W && patchH * (patchW + 1) <= 128)
    patchW++;
  while(patchH < H && (patchH + 1) * patchW <= 128)
    patchH++;

  // We are assuming that the thread-block size can be as large as 512; this
  // works even on old CUDA architectures.
  int threads_per_pixel;
  if (patchH * patchW * 4 <= 512 && (kH * kW) > 8)
    threads_per_pixel = 4;
  else if (patchH * patchW * 2 <= 512 && (kH * kW) > 4)
    threads_per_pixel = 2;
  else
    threads_per_pixel = 1;

  int threads_per_block = patchH * patchW * threads_per_pixel;
  // round threads_per_block up to a multiple of 64.  We need it to be
  // equivalent to an even number of warps, because at one point we divide the
  // threads into two halves and we want them to be an even number of warps.
  threads_per_block = 64 * ((threads_per_block + 63) / 64);

  {
    // If it's possible to increase the patch width or height while not exceeding
    // this number of threads, do so.  (This is a small optimization).
    int patchW_old = patchW;
    while (patchH * (patchW + 1) * threads_per_pixel <= threads_per_block)
      patchW++;
    // If the above change to patchW did not actually reduce the number of patches
    // needed to cover the image, gthen there is no point to the change; and it
    // increases the shared-memory requirement, so revert it.
    if ((W + patchW_old - 1) / patchW_old == (W + patchW - 1) / patchW)
      patchW = patchW_old;
    int patchH_old = patchH;
    while ((patchH + 1) * patchW * threads_per_pixel <= threads_per_block)
      patchH++;
    if ((H + patchH_old - 1) / patchH_old == (H + patchH - 1) / patchH)
      patchH = patchH_old;
  }


  int threads_per_kernel_pos = 1;
  while (threads_per_kernel_pos < 32 &&
         threads_per_kernel_pos * 2 * kH * kW <= threads_per_block)
    threads_per_kernel_pos *= 2;

  // dimensions of padded patches
  int ppatchH = patchH + kH - 1,
       ppatchW = patchW + kW - 1,
   ppatch_size = ppatchH * ppatchW;

  int buffer_numel = 4 * (kH * kW) + 3 * ppatch_size + threads_per_block;

  int num_patches_H = (H + patchH - 1) / patchH,
      num_patches_W = (W + patchW - 1) / patchW,
      num_patches = num_patches_H * num_patches_W;

  // gridDim.x == C.
  int num_blocks_patch = 1,  // gridDim.y.  should not be more
      num_blocks_batch = 1;  // gridDim.z
  // We have a rough target of no more than 256 thread-groups.
  while (C * num_blocks_patch * 2 <= 256 &&
         num_blocks_patch * 2 <= num_patches)
    num_blocks_patch *= 2;
  if (C * num_patches <= 512)
    num_blocks_patch = num_patches;
  while (C * num_blocks_patch * num_blocks_batch * 2 <= 256 &&
         num_blocks_batch * 2 <= N)
    num_blocks_batch *= 2;

  assert(num_blocks_patch <= num_patches && num_blocks_batch <= N);
  assert(patchH * patchW * threads_per_pixel <= threads_per_block);
  assert(kH * kW * threads_per_kernel_pos <= threads_per_block);

  static int debug_count = 50;
  if (debug_count > 0) {
    debug_count--;
    std::cout << "[backward:] N,C,H,W=" << N << "," << C << "," << H << "," << W
              << "; kW,kH=" << kW << "," << kH
              << "; patchH,patchW=" << patchH << ","
              << patchW << ", num_blocks_patch="
              << num_blocks_patch << ", num_blocks_batch="
              << num_blocks_batch
              << ", threads_per_pixel=" << threads_per_pixel
              << ", threads_per_kernel_pos=" << threads_per_kernel_pos
              << ", threads_per_block=" << threads_per_block
              << ", buffer_numel=" << buffer_numel
              << std::endl;
  }

  int num_blocks = num_blocks_patch * num_blocks_batch;

  torch::Tensor grad_input = torch::zeros({N, 2*C, H, W},
                                          torch::TensorOptions().dtype(scalar_t).device(input.device())),
      grad_pos_add = torch::zeros({num_blocks, C, kH, kW},
                                  torch::TensorOptions().dtype(scalar_t).device(input.device())),
      grad_pos_mul = torch::zeros({num_blocks, C, kH, kW},
                                  torch::TensorOptions().dtype(scalar_t).device(input.device()));


  dim3 gridDim(C, num_blocks_patch, num_blocks_batch);
  // blockDim is scalar, just threads_per_block.
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "learned_nonlin_kernel_backward", ([&] {
        learned_nonlin_kernel_backward<scalar_t><<<gridDim, threads_per_block,
                                                    sizeof(scalar_t) * buffer_numel,
                                                    at::cuda::getCurrentCUDAStream()>>>(
              input.packed_accessor32<scalar_t, 4>(),
              pos_add.packed_accessor32<scalar_t, 3>(),
              pos_mul.packed_accessor32<scalar_t, 3>(),
              grad_output.packed_accessor32<scalar_t, 4>(),
              grad_input.packed_accessor32<scalar_t, 4>(),
              grad_pos_add.packed_accessor32<scalar_t, 4>(),
              grad_pos_mul.packed_accessor32<scalar_t, 4>(),
              patchH,
              patchW,
              threads_per_pixel,
              threads_per_kernel_pos);
      }));
  grad_pos_add = at::sum(grad_pos_add, {0});
  grad_pos_mul = at::sum(grad_pos_mul, {0});

  return std::vector<torch::Tensor>({grad_input, grad_pos_add, grad_pos_mul}); */
}
