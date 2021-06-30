#include <torch/extension.h>
#include <cooperative_groups.h>




template <typename scalar_t, typename group_t>
__device__ int reduce_sum(group_t g, scalar_t *temp, scalar_t val)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    #pragma unroll
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if (lane < i) val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }

    return val; // note: only thread 0 will return full sum
}


/*
  Forward of integrated_conv.  Each thread group handles a single channel
  (equal to blockIdx.x), and loops over patches of the output.

  Template args:
      scalar_t: the floating-point type, e.g. float, double, maybe half.
      buffer_dim:  The number of scalar_t in the shared-memory buffer; this is
                shared between the input patch and pieces of pos_add
                and pos_mul.  It is user's responsibility to ensure that
                buffer_dim is large enough for the provided parameters.

  Args:
      input:  input image, shape (N, 2*C, H, W)
      pos_add:  positional encoding, additive part,  shape (C, kH, kW)
      pos_add:  positional encoding, multiplicative part, shape (C, kH, kW)
   Note: kH and kW must both be odd so that it's clear how to pad.

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
    torch::PackedTensorAcessor32<scalar_t, 4> output,  // N, C, H, W
    int opatchH,  // output-patch height,
    int opatchW  // output-patch width
                             ) {
  const int H = input.size(2),
      W = input.size(3)
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
      ipatchW = ipatchW + kW - 1,
      ipatch_size = ipatchH * ipatchW,
      opatch_size = opatchH * opatchW;

  // `extern_buf` is general-purpose shared memory, which we'll divide between
  // pos_add, pos_mul and src_img_buf to be shared between the src image size
  // (ipatch_size) and the number of threads (blockDim.x)
  __shared__ scalar_t buf[buffer_dim];

  __shared__ scalar_t
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


  threads_per_opixel = blockDim.x / opatch_size;
  assert(blockDim.x == opatch_size * threads_per_opixel);

  auto tile = cooperative_groups::tiled_partition(g, threads_per_opixel);

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
        if ((unsigned int)src_h < (unsigned int)H &&
            (unsigned int)src_w < (unsigned int)W)
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
        dest_val = input[n][c + C][h][w];  // else 0.
      }

      // `sum` is the partial sum that this thread computes; we'll sum this over
      // the `threads_per_opixel` threads in the tile to get the output pixel
      // value.
      scalar_t sum = 0.0;

      for (int pos_in_kernel = tile.thread_rank();
           pos_in_kernel < (kH * kW);
           pos_in_kernel += threads_per_opixel) {
        int h_in_kernel = pos_in_kernel / kW,
            w_in_kernel = pos_in_kernel % kW;
        // Note: this is actually more like cross-correlation, as we don't
        // have a negative sign on the h and w indexes in the kernel.
        // Also note: we already took care of padding and the associated
        // offsets of -(kH / 2) and -(kW / 2).
        int h_in_src_patch = h_in_patch + h_in_kernel,
            w_in_src_patch = w_in_patch + w_in_kernel;
        scalar_t src_val = src_img_buf[h_in_src_patch * ipatchW + w_in_src_patch],
            pos_add_val = pos_add_buf[pos_in_kernel];
        scalar_t relu = (src_val + dest_val + pos_add_val);
        if (relu > 0.0)
          sum += relu * pos_mul_buf[pos_in_kernel];
      }
      // Aggregate `sum` over threads, if needed; and write the result to `output`.
      if (threads_per_opixel > 1) {
        __syncthreads();
        src_img_buf[threadIdx.x] = sum;
        __syncthreads();
        if (tile.thread_rank() == 0 && h < H && w < W) {
          // This linear summation should be OK because threads_per_opixel is
          // unlikely to be greater than 4.
          for (int i = 1; i < threads_per_opixel; i++)
            sum += src_img_buf[threadIdx.x + i];
          output[n][c][h][w] = sum;
        }
      } else {
        if (h < H && w < W)
          output[n][c][h][w] = sum;
      }
    }
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
  dtype scalar_t = input.dtype();
  TORCH_CHECK(pos_add.dtype() == scalar_t &&
              pos_mul.dtype() == scalar_t,
              "Input dtypes mismatch");

  torch::Tensor output = torch::empty({N, C, H, W},
                                      torch::TensorOptions().dtype(scalar_t).device(input.device()));


  // Work out the configuration with which we call the kernel..

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

  // We are assuming that the thread-block size can be as large as 1024; this is
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

  int buffer_numel = 2 * (kH * kW) + max<int>(threads_per_block,
                                              input_patch_size);


  int num_patches_H = (H + patchH - 1) / patchH,
      num_patches_W = (W + patchW - 1) / patchW,
      num_patches = num_patches_H * num_patches_W;

  // gridDim.x == C.
  int num_blocks_patch = 1,  // gridDim.y.  should not be more
       num_blocks_batch = 1;
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
            << num_blocks_batch << std::endl;


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
  return std::vector<torch::Tensor>();
}
