#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // for getCurrentCUDAStream()
#include <cooperative_groups.h>




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
  Forward of integrated_conv.  Each thread group handles a single channel (equal
  to blockIdx.x), and loops over patches of the output and over the image n
  within the batch (different thread groups may be responsible for different
  subsets of patches and/or images, see docs of gridDim below).

  Template args:
      scalar_t: the floating-point type, e.g. float, double, maybe half.

  Args:
      input:  input image, shape (N, 2*C, H, W)
      pos_add:  positional encoding, additive part,  shape (C, kH, kW)
      pos_mul:  positional encoding, multiplicative part, shape (C, kH, kW)
      output:   output image, shape (N, 2*C, H, W)
   Note: kH and kW must both be odd so that it's clear how to zero-pad.

  The thread-block should have one dimension (x); blockDim.x should equal
  some small power of 2 (threads_per_opixel) times the output-patch size which is
  opatchH * opatchW (the output-patch height and width).  We expect
  threads_per_opixel to be 1, 2, or 4; we use a linear summation to sum up the
  different threads' partial sums, and if threads_per_opixel gets larger we'd
  need to make this a logarithmic reduction.

   The requirements on the grid dimension are:
       gridDim.x == num-channels C (required)
       gridDim.y <= num-patches per image (recommended)
       gridDim.z <= batch-size N (recommended)
  When we invoke this kernel, we'll invoke it as:
   integrated_conv_forward<<<gridDim, blockDim, bytesShared, stream>>>
  where bytesShared is the number of bytes needed in `extern_buf`:
    bytesShared = sizeof(shared_t) * numel, where
    numel = 2 * (kH * kW) + max(blockDim.x, (opatchH + kH - 1) * (patchW + kW - 1))
 */
extern __shared__ int extern_buf[];

template <typename scalar_t>
__global__
void integrated_conv_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4> input,  // N, 2*C, H, W
    torch::PackedTensorAccessor32<scalar_t, 3> pos_add,  // C, kH, kW
    torch::PackedTensorAccessor32<scalar_t, 3> pos_mul,  // C, kH, kW
    torch::PackedTensorAccessor32<scalar_t, 4> output,  // N, C, H, W
    int opatchH,  // output-patch height,
    int opatchW  // output-patch width
                             ) {
  const int H = input.size(2),
      W = input.size(3),
      kH = pos_add.size(1),
      kW = pos_add.size(2),
      npatchH = (H + opatchH - 1) / opatchH,  // num patches in vertical dim
      npatchW = (W + opatchW - 1) / opatchW,  // num patches in horizontal dim
      npatch = npatchH * npatchW;  // total number of patches per image

  // Channel index.
  const int c = blockIdx.x;
  // We don't need to check the range of `c` because we set gridDim.x to the
  // exact number of channels.

  const int ipatchH = opatchH + kH - 1,
      ipatchW = opatchW + kW - 1,
      ipatch_size = ipatchH * ipatchW,
      opatch_size = opatchH * opatchW;

  // `extern_buf` is general-purpose shared memory, which we'll divide between
  // pos_add, pos_mul and src_img_buf to be shared between the src image size
  // (ipatch_size) and the number of threads (blockDim.x)

  // these are pointers to __shared__ memory; the compiler should
  // be able to figure this out.
  scalar_t
      *pos_add_buf = (scalar_t*)extern_buf,     // pos_add positional-encoding / kernel parameters,
                                // indexed [kh*kW + kw] where kh and kw are vertical
                                // and horizontal positions in the kernel.
      *pos_mul_buf = pos_add_buf + (kH * kW), // pos_mul positional-encoding / kernel parameters,
                                              // indexed [kh*kW + kw] where kh and kw are vertical
                                              // and horizontal positions in the kernel.
      *src_img_buf = pos_mul_buf + (kH * kW);    // version of input image that relates to source position,
                             // of size [ipatch_size], indexed [h*ipatchW + w]...
                             // note, the 'h' and 'w' indexes are into the zero-padded input
                             // image.


  int threads_per_opixel = blockDim.x / opatch_size;
  assert(blockDim.x == opatch_size * threads_per_opixel);

  // pos_in_patch will be interpreted as h_in_patch * opatchW + w_in_patch.
  int pos_in_patch = threadIdx.x / threads_per_opixel;

  // Load parts of the kernel parameters pos_add and pos_mul into shared memory,
  // in pos_add_buf and pos_mul_buf
  for (int i = threadIdx.x; i < kH * kW; i += blockDim.x) {
    int kh = i / kW,
        kw = i % kW;
    pos_add_buf[i] = pos_add[c][kh][kw];
    pos_mul_buf[i] = pos_mul[c][kh][kw];
  }

  // n is the index within the batch.  Loop to make sure we cover all images in
  // the batch.  input.size(0) is the batch size N.  All threads in the thread-block
  // loop the same number of times.
  for (int n = blockIdx.z; n < input.size(0); n += gridDim.z) {

    // Loop over the patch within the output image.  All threads in the
    // thread-block loop the same number of times.
    for (int patch_idx = blockIdx.y; patch_idx < npatch; patch_idx += gridDim.y) {
      // (patch_h_offset, patch_w_offset) are the (vertical, horizontal) indexes
      // of the lowest-numbered pixel in the patch of output that this thread
      // block is responsible for.
      int patch_h_offset = (patch_idx / npatchW) * opatchH,
          patch_w_offset = (patch_idx % npatchW) * opatchW;

      // This __syncthreads() is only necessary if we have already looped at
      // least once over n or patch_idx: it's in case other threads are still
      // using the `src_img_buf` buffer for something else.
      __syncthreads();

      // Load the 'src' part of the input patch; the size of this is the size of
      // the output patch plus a border of sizes kH//2, kW//2 on each side.
      for (int i = threadIdx.x; i < ipatch_size; i += blockDim.x) {
        int h_in_kernel = i / ipatchW,
            w_in_kernel = i % ipatchW;
        int src_h = patch_h_offset + h_in_kernel - (kH / 2),  // kH / 2 is offset due to padding
            src_w = patch_w_offset + w_in_kernel - (kW / 2);
        scalar_t src_val = scalar_t(0);
        if ((unsigned int)src_h < (unsigned int)H &&  // h >= 0 && h < H
            (unsigned int)src_w < (unsigned int)W)    // w >= 0 && w < W
          src_val = input[n][c][src_h][src_w];
        src_img_buf[i] = src_val;
      }
      // make sure all threads have written to `src_img_buf`
      __syncthreads();


      // 'h' and 'w' are the positions within the output image, that this tile
      // of size threads_per_opixel is responsible for.
      int h = patch_h_offset + pos_in_patch / opatchW,
          w = patch_w_offset + pos_in_patch % opatchW;

      // The "destination" pixel; this is an input.  It gets added to each
      // src pixel, prior to the relu, in the loop below.
      scalar_t dest_val = scalar_t(0);
      if (h < H && w < W) {
        // Several threads (within the same tile, which implies the same warp)
        // may load the same value here, but I believe the device's memory
        // subsystem handles this well enough that we can just ignore the issue
        // rather than try to optimize it.
        // https://forums.developer.nvidia.com/t/accessing-same-global-memory-address-within-warps/66574
        int C = input.size(1) / 2;
        dest_val = input[n][c + C][h][w];  // else 0.
      }

      // `sum` is the partial sum that this thread computes; we'll sum this over
      // the `threads_per_opixel` threads in the tile to get the output pixel
      // value.
      scalar_t sum = 0.0;

      for (int pos_in_kernel = threadIdx.x % threads_per_opixel;
           pos_in_kernel < (kH * kW);
           pos_in_kernel += threads_per_opixel) {
        int h_in_kernel = pos_in_kernel / kW,
            w_in_kernel = pos_in_kernel % kW;
        // Note: this is actually more like cross-correlation, as we don't
        // have a negative sign on the h and w indexes in the kernel.
        // Also note: we already took care of padding and the associated
        // offsets of -(kH / 2) and -(kW / 2).
        int h_in_src_patch = (pos_in_patch / opatchW) + h_in_kernel,
            w_in_src_patch = (pos_in_patch % opatchW) + w_in_kernel;
        scalar_t src_val = src_img_buf[h_in_src_patch * ipatchW + w_in_src_patch],
            pos_add_val = pos_add_buf[pos_in_kernel];
        scalar_t relu = (src_val + dest_val + pos_add_val);
        if (relu > 0.0)
          sum += relu * pos_mul_buf[pos_in_kernel];
      }
      // Aggregate `sum` over threads
      sum = tiled_warp_reduce_sum(threads_per_opixel, src_img_buf, sum);
      if (threadIdx.x % threads_per_opixel == 0 && h < H && w < W) {
        output[n][c][h][w] = sum;
      }
    }
  }
}


/*
  Backward of integrated_conv.  Each thread group handles a single channel (equal
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
   integrated_conv_forward<<<gridDim, blockDim, bytesShared, stream>>>
  where bytesShared is the number of bytes needed in `extern_buf`:

   bytesShared = sizeof(shared_t) * numel, where
    numel = 4 * (kH * kW) + 3 * (ppatchH * ppatchW) + blockDim.x
 */


template <typename scalar_t>
__global__
void integrated_conv_kernel_backward(
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
      *grad_output_buf = src_img_buf + ppatch_size, // output gradient for padded patch, indexed [h*ppatchW + w]
      *grad_pos_add_buf = grad_output_buf + ppatch_size,  // total grad for pos_add for this thread block, indexed [kh*kW + kw]
      *grad_pos_mul_buf = grad_pos_add_buf + (kH * kW),  // total grad for pos_mul for this thread block, indexed [kh*kW + kw]
      *reduce_buf = grad_pos_mul_buf + (kH * kW);  // buffer for reduction over threads, size == blockDim.x


  // pos_in_patch will be interpreted as h_in_patch * patchW + w_in_patch.
  int pos_in_patch = threadIdx.x / threads_per_pixel;

  // Load parts of the kernel parameters pos_add and pos_mul into shared memory,
  // in pos_add_buf and pos_mul_buf; zero the corresponding gradient buffers.
  // We know that blockDim.x >= kH * kW, see threads_per_kernel_pos.
  if (threadIdx.x < kH * kW) {
    int i = threadIdx.x;
    int kh = i / kW, kw = i % kW;
    pos_add_buf[i] = pos_add[c][kh][kw];
    pos_mul_buf[i] = pos_mul[c][kh][kw];
    grad_pos_add_buf[i] = 0.0;
    grad_pos_mul_buf[i] = 0.0;
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
      for (int i = threadIdx.x % (blockDim.x / 2); i < ppatch_size; i += (blockDim.x / 2)) {
        int h_in_ppatch = i / ppatchW,
            w_in_ppatch = i % ppatchW;
        int h = patch_h_offset + h_in_ppatch - (kH / 2),  // kH / 2 is offset due to padding
            w = patch_w_offset + w_in_ppatch - (kW / 2);

        if (threadIdx.x < blockDim.x / 2) {  // The first half of the threads of the block
                                             // load `input`
          scalar_t src_val = scalar_t(0),
              dest_val = scalar_t(0);
          if ((unsigned int)h < (unsigned int)H &&  // h >= 0 && h < H.
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
        // This block computes `grad_input_sum`.
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
          // To compute a contribution to "this_input_src_grad", we need to consider the
          // contribution to the destination pixel that it would have contributed to
          // with this same offset.
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
        // for position (kh, kw) in the kernel; we iterate over the patch.
        scalar_t pos_add_val = pos_add_buf[pos_in_kernel],
            pos_mul_val = = pos_mul_buf[pos_in_kernel];

        for (int pos_in_patch = threadIdx.x % threads_per_kernel_pos;
             pos_in_patch < patch_size; pos_in_patch += threads_per_kernel_pos) {
          // We are working out the contribution to the gradients for pos_add
          // and pos_mul; we let `pos_in_patch` correspond to the *output*
          // position, and work out the input position based on gthe kernel position.

          int h_in_patch = pos_in_patch / patchH,
              w_in_patch = pos_in_patch / patchW;

          // pos_in_ppatch is the position in the padded patch corresponding to
          // `pos_in_patch`.
          int pos_in_ppatch = (h_in_patch + kH / 2) * ppatchW + (w_in_patch + kW / 2);
          scalar_t dest_val = dest_img_buf[pos_in_ppatch];
          int offset_pos_in_ppatch = (h_in_patch + kh) * ppatchW + (w_in_patch + kw);
          scalar_t src_val = src_img_buf[offset_pos_in_ppatch];

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
          grad_pos_add_buf[pos_in_kernel] = this_grad_pos_add;
          grad_pos_mul_buf[pos_in_kernel] = this_grad_pos_mul;
        }
      }
    }
  }

  int block = blockIdx.z * gridDim.y + blockIdx.y;

  int kernel_pos = threadIdx.x;
  if (kernel_pos < (kH * kW)) {
    int kh = kernel_pos / kW,
        kw = kernel_pos % kW;
    grad_pos_add[block][c][kh][kw] = grad_pos_add_buf[kernel_pos];
    grad_pos_mul[block][c][kh][kw] = grad_pos_mul_buf[kernel_pos];
  }
}








torch::Tensor integrated_conv_cuda(torch::Tensor input,
                                   torch::Tensor pos_add,
                                   torch::Tensor pos_mul) {
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

  torch::Tensor output = torch::empty({N, C, H, W},
                                      torch::TensorOptions().dtype(scalar_t).device(input.device()));


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
  int threads_per_opixel;
  if (patchH * patchW * 4 <= 512 && (kH * kW) > 16)
    threads_per_opixel = 4;
  else if (patchH * patchW * 2 <= 512 && (kH * kW) > 8)
    threads_per_opixel = 2;
  else
    threads_per_opixel = 1;

  int input_patchH = patchH + kH - 1,
         input_patchW = patchW + kW - 1,
         input_patch_size = input_patchH * input_patchW;

  int threads_per_block = patchH * patchW * threads_per_opixel;

  int buffer_numel = 2 * (kH * kW) + std::max<int>(threads_per_block,
                                                   input_patch_size);

  int num_patches_H = (H + patchH - 1) / patchH,
      num_patches_W = (W + patchW - 1) / patchW,
      num_patches = num_patches_H * num_patches_W;

  // gridDim.x == C.
  int num_blocks_patch = 1,  // gridDim.y.
       num_blocks_batch = 1;  // gridDim.z
  while (C * num_blocks_patch <= 256 &&
         num_blocks_patch * 2 <= num_patches)
    num_blocks_patch *= 2;
  if (C * num_patches <= 512)
    num_blocks_patch = num_patches;
  while (C * num_blocks_patch * num_blocks_batch <= 512 &&
         num_blocks_batch * 2 <= N)
    num_blocks_batch *= 2;
  if (C * num_blocks_patch * N <= 1024)
    num_blocks_batch = N;

  assert(num_blocks_patch <= num_patches && num_blocks_batch <= N);

  std::cout << "N,C,H,W=" << N << "," << C << "," << H << "," << W
            << "; kW,kH=" << kW << "," << kH
            << "; patchH,patchW=" << patchH << ","
            << patchW << ", num_blocks_patch="
            << num_blocks_patch << ", num_blocks_batch="
            << num_blocks_batch
            << ", threads_per_opixel=" << threads_per_opixel
            << ", threads_per_block=" << threads_per_block
            << std::endl;

  dim3 gridDim(C, num_blocks_patch, num_blocks_batch);
  // blockDim is scalar, just threads_per_block.
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "integrated_conv_kernel", ([&] {
        integrated_conv_kernel<scalar_t><<<gridDim, threads_per_block, sizeof(scalar_t) * buffer_numel, at::cuda::getCurrentCUDAStream()>>>(
              input.packed_accessor32<scalar_t, 4>(),
              pos_add.packed_accessor32<scalar_t, 3>(),
              pos_mul.packed_accessor32<scalar_t, 3>(),
              output.packed_accessor32<scalar_t, 4>(),
              patchH,
              patchW);
      }));
  return output;
}



std::vector<torch::Tensor> integrated_conv_backward_cuda(torch::Tensor input,
                                                         torch::Tensor pos_add,
                                                         torch::Tensor pos_mul,
                                                         torch::Tensor grad_output) {
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

  int num_blocks = num_blocks_patch * num_blocks_batch;

  torch::Tensor grad_input = torch::zeros({N, 2*C, H, W},
                                          torch::TensorOptions().dtype(scalar_t).device(input.device())),
      grad_pos_add = torch::zeros({num_blocks, C, kH, kW},
                                  torch::TensorOptions().dtype(scalar_t).device(input.device())),
      grad_pos_mul = torch::zeros({num_blocks, C, kH, kW},
                                  torch::TensorOptions().dtype(scalar_t).device(input.device()));


  dim3 gridDim(C, num_blocks_patch, num_blocks_batch);
  // blockDim is scalar, just threads_per_block.
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "integrated_conv_kernel", ([&] {
        integrated_conv_kernel<scalar_t><<<gridDim, threads_per_block, sizeof(scalar_t) * buffer_numel, at::cuda::getCurrentCUDAStream()>>>(
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

  return std::vector<torch::Tensor>({grad_input, grad_pos_add, grad_pos_mul});
}
