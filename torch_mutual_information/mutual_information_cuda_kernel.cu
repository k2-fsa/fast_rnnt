#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // for getCurrentCUDAStream()
#include <cooperative_groups.h>
#include <cmath>  // for INFINITY



// returns log(exp(x) + exp(y)).
__forceinline__ __device__ double LogAdd(double x, double y) {
  double diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.
  if (diff - diff != 0)
    return x;    // x and y are probably -inf.  Return the larger one.
  else
    return x + log1p(exp(diff));
}

// returns log(exp(x) + exp(y)).
__forceinline__ __device__ inline float LogAdd(float x, float y) {
  float diff;
  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.
  if (diff - diff != 0)
    return x;    // x and y are probably -inf.  Return the larger one.
  else
    return x + log1p(exp(diff));
}




/*
  Forward of mutual_information.  Each thread block handles blocks of (x, y) shape
  equal to (BLOCK_SIZE, BLOCK_SIZE), e.g. (32, 32).  Thread blocks loop over such
  blocks, but they might loop only once if there is not that much data to process.
  We sequentially launch groups of threads in such a way that thread-blocks
  within a group do not depend on each other.


  Template args:
      scalar_t: the floating-point type, e.g. float, double, maybe half.

  Args:
      px:     log-odds ratio of generating next x in the sequence, i.e.
              xy[b][s][t] is the log-odds probability of generating x_t of
              the b'th image given subsequences of length (s, t).  (See
              mutual_information.py for more info).  Shape [B][S][T + 1]
      py:     log-odds ratio of generating next y in the sequence.
              Shape [B][S + 1][T]
      p:      matrix of mutual information of sub-sequences, that this
              function writes to.  Shape [B][S + 1][T + 1].  This function
              computes the following recursion:

               p[b,0,0] = 0.0
               p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                  p[b,s,t-1] + py[b,s,t-1])
                       (if s > 0 or t > 0)


   boundary:  If set, a tensor of shape [B][4] of type int64_t, which
              contains, for each batch element, [s_begin, t_begin, s_end, t_end]
              which are the beginning and end (one-past-the-last) of the
              x and y sequences that we should process.  If not set, these
              default to (0, 0, S, T), and they should not exceed these bounds
              or be empty (i.e. s_begin <= t_begin or s_end <= t_end).


              nput:  input image, shape (B, C, T) where B is batch size, C is
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

   The block-dim and grid-dim must both be 1-dimensional, and the block-dim must
   be at least 128.
 */


template <typename scalar_t,
          int BLOCK_SIZE>   // e.g. BLOCK_SIZE == 16 or 32.   Note: we require the
                            // num-threads be at least 128.
__global__
void mutual_information_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3> px,   // B, S, T + 1, i.e. batch, x_seq_length, y_seq_length + 1
    torch::PackedTensorAccessor32<scalar_t, 3> py,   // B, S + 1, T.
    torch::PackedTensorAccessor32<scalar_t, 3> p,    // B, S + 1, T + 1.  This is an output.
    torch::PackedTensorAccessor32<int64_t, 2> boundary,  // B, 4;  or 0, 0 if boundaries are the defaults (0, 0, S, T)
    torch::PackedTensorAccessor32<scalar_t, 1> ans,  // [B]
    int iter) {  // This kernel is sequentially called with 'iter' = 0, 1, 2 and so on,
                 // up to num_iters - 1 where
                 // num_iters = num_s_blocks + num_t_blocks - 1
                 // num_s_blocks = S / BLOCK_SIZE + 1
                 // num_t_blocks = T / BLOCK_SIZE + 1
                 // so that each group depends on the previous group...
  const int B = px.size(0),
      S = px.size(1),
      T = py.size(2);
  // num_s_blocks and num_t_blocks are the number of blocks we need to cover the
  // array of size (S, T) with blocks of this size, in the s and t directions
  // respectively.
  // You can read the following expressions as simplifications of, for example,
  // num_s_blocks = ((S + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE,
  // i.e. rounding-up division of (S + 1) by BLOCK_SIZE, and the same for (T + 1).
  const int num_s_blocks = S / BLOCK_SIZE + 1,
      num_t_blocks = T / BLOCK_SIZE + 1;

  // num_blocks_this_iter is an upper bound on the number of blocks of size
  // (BLOCK_SIZE by BLOCK_SIZE) that might be active on this iteration.  We go
  // from the bottom left of the image so that on iter == 0 we process only one
  // block with block-index (0, 0) then on iter == 1 we process block-indexes
  // (1, 0) and (0, 1); and then on iter==2 we process (2, 0), (1, 1) and (0,
  // 2); and so on.  We also will never have more than `num_s_blocks` blocks
  // (We'll never have more than num_t_blocks either, but the numbering we use
  // corresponds to s and not t, so if we hit the num_t_blocks limit, the
  // lowest-numbered blocks on s would just not be active and we'll 'continue'
  // below).
  int num_blocks_this_iter = min(iter + 1, num_s_blocks);


  // For the block with s_block_begin == 0 and t_block_begin == 0 (for
  // easy illustration), px_buf[s][t] will contain exp(px[s - 1][t]); or 0
  // for out-of-range indexes.
  // Likewise, py_buf[s][t] will contain exp(py[s][t - 1]).
  __shared__ scalar_t px_buf[BLOCK_SIZE][BLOCK_SIZE],
      py_buf[BLOCK_SIZE][BLOCK_SIZE];

  // 1st row/col of p_buf correspond to the previous blocks, or to an edge case.
  // So, again for this origin block, p_buf[s][t] corresponds to exp(p[s - 1][t
  // - 1] - normalizer); or 0 for out-of-range values.
  __shared__ scalar_t p_buf[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

  // boundary_buf will be used to store the b'th row of `boundary` if we have
  // boundary information supplied.
  __shared__ int64_t boundary_buf[4];

  if (threadIdx.x == 0) {
    boundary_buf[0] = 0;
    boundary_buf[1] = 0;
    boundary_buf[2] = S;
    boundary_buf[3] = T;
  }

  // batch_block_iter iterates over both batch elements (index b), and block
  // indexes in the range [0..num_blocks_this_iter-1]
  for (int batch_block_iter = blockIdx.x;
       batch_block_iter < B * num_blocks_this_iter;
       batch_block_iter += gridDim.x) {
    int b = batch_block_iter % B,
        block = batch_block_iter / B;

    int s_block_begin = block * BLOCK_S_SIZE,
        t_block_begin = (iter  - block) * BLOCK_T_SIZE;


    if (threadDim.x < 4 && boundary.size(0) != 0)
      boundary_buf[threadDim.x] = boundary[b][threadDim.x];
    __syncthreads();
    int s_begin = boundary_buf[0],
        t_begin = boundary_buf[1],
        s_end = boundary_buf[2],
        t_end = boundary_buf[3];
    s_block_begin += s_begin;
    t_block_begin += t_begin;

    // block_S and block_T are the actual sizes of this block, no greater than
    // (BLOCK_SIZE, BLOCK_SIZE) but possibly less than that if we are towards
    // the end of the sequence.
    // The last element of the output matrix p we write is (s_end, t_end),
    // i.e. the one-past-the-end index is (s_end + 1, t_end + 1).
    int block_S = min(BLOCK_SIZE, s_end + 1 - s_block_begin),
        block_T = min(BLOCK_SIZE, t_end + 1 - t_block_begin);

    if (block_S <= 0 || block_T <= 0)
      continue;

    bool is_origin_block = (s_block_begin * t_block_begin == 0);

    // Load px_buf and py_buf.  We exponentiate; the assumption is that they most likely
    // won't overflow or underflow, but if they do overflow we'll detect it later; we'll
    // also detect certain kinds of underflow.
    for (int i = threadDim.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE,
          t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin,
          t = t_in_block + t_block_begin;

      // the comparisons with S and T below just make sure we don't access
      // out-of-memory regions; they do not guarantee we are in the range given
      // by s_begin, s_end and so on.  Note: comparing as unsigned int makes sure
      // the index is nonnegative.
      scalar_t this_px = 0.0;
      if (static_cast<unsigned int>(s - 1) < static_cast<unsigned int>(S) &&
          t <= T)
        this_px = exp(px[b][s - 1][t]);
      px_buf[s_in_block][t_in_block] = this_px;
      scalar_t this_py = 0.0;
      if (static_cast<unsigned int>(t - 1) < static_cast<unsigned int>(T) &&
          s <= S)
        this_py = exp(py[b][s][t - 1]);
      py_buf[s_in_block][t_in_block] = this_py;
    }


    // Load the 1st row and column of p_buf (except element[0][0] is not needed).
    // Remember: p_buf[s][t] corresponds to exp(p[s + s_block_begin - 1][t + t_block_begin - 1] - normalizer.
    if (threadIdx.x < 64) {  // 64 == warp size.  First half of threads...
      if (threadIdx.x <= BLOCK_SIZE) {
        // s_in_p_buf are simply the indexes into p_buf
        int s_in_p_buf = threadIdx.x,
            t_in_p_buf = 0,
            s = s_in_p_buf + s_block_begin - 1,
            t = t_in_p_buf + t_block_begin - 1;
        // The if-statement below just guards against out-of-range memory
        // accesses, it does not guarantee that we really need these values.
        scalar_t this_p = -INFINITY;
        if (static_cast<unsigned int>(s) < static_cast<unsigned int>(S) &&
            static_cast<unsigned int>(t) < static_cast<unsigned int>(T))
          this_p = p[s + s_block_begin][s + t_block_begin];
        p_buf[threadIdx.x][0] = this_p;
      }
    } else { // Another warp handles the other leg
      if (threadIdx.x - 64 <= BLOCK_SIZE) {
        int s_in_p_buf = 0,
            t_in_p_buf = threadIdx.x - 64,
            s = s_in_p_buf + s_block_begin - 1,
            t = t_in_p_buf + t_block_begin - 1;
        // The if-statement below just guards against out-of-range memory
        // accesses, it does not guarantee that we really need these values.
        scalar_t this_p = -INFINITY;
        if (static_cast<unsigned int>(s) < static_cast<unsigned int>(S) &&
            static_cast<unsigned int>(t) < static_cast<unsigned int>(T))
          this_p = p[s + s_block_begin][s + t_block_begin];
        p_buf[threadIdx.x][0] = this_p;
      }
    }

    __syncthreads();

    // We read p_buf in log-space; subtract 'normalizer', which mathematically
    // could be any finite number, to get in a reasonable range of probabilities,
    // and then exponentiate.  We'll do everything in non-log space, and later
    // take a log before we write out the data.
    scalar_t normalizer = (is_origin_block ? 0.0 :
                           max(px_buf[0][1], px_buf[1][0]));

    // Normalize and exponentiate the edge elements of p_buf, i.e. the elements
    // where at one index is 0.  The [0][0] element is special; we write 0.0,
    // and we'll overwrite with 1.0 if there is a panic situation due to
    // overflow.
    if (threadIdx.x <= BLOCK_SIZE) {
      if (threadIdx.x == 0) {
        // p_buf[0][0] is never used for its normal purpose; we set it to zero.
        // We'll later write an infinity there if something goes wrong, as a
        // 'panic' indicator.
        p_buf[threadIdx.x][0] = (threadIdx.x == 0 ? 0.0 :
                                 exp(p_buf[threadIdx.x][0] - normalizer));
      }
    } else if ((int)threadIdx.x - 64 < BLOCK_SIZE) {
      p_buf[0][threadIdx.x + 1] = exp(p_buf[0][threadIdx.x + 1] - normalizer);
    }


    if (threadIdx.x == 0) {
      // This if-statement is an optimization and modification of the loop below
      // for the value i == 0, i.e. inner-iteration == 0.  The modification
      // is to use 0.0 if this is the "origin block" with s_block_begin == 0 and
      // t_block_begin == 0.  This corresponds to the probability of the pair of
      // sequences of length (0, 0).
      p_buf[1][1] = (is_origin_block ? 0.0 :
                     p_buf[0][1] * px_buf[0][0] +
                     p_buf[1][0] * py_buf[0][0]);
    }

    scalar_t p_buf_s1_t;  // This is for an optimization.
    if (i < BLOCK_SIZE) {
      int s = threadIdx.x;
      p_buf_s1_t = p_buf[s + 1][0];
    }

    for (int i = 1; i < block_S + block_T; i++) {
      // i is the inner iteration, which corresponds to the (s + t) indexes of the
      // elements within the block that we write.  So i == 0 writes positions
      // (s, t) == (0, 0); i == 1 writes (0, 1) and (1, 0); i == 2 writes
      // (0, 2), (1, 1) and (2, 1); and so on.
      // Note: not many threads participate in this part, only up to BLOCK_SIZE
      // at most.  Unfortunately we couldn't figure out a very meaningful way
      // for more threads to do work, that looked like it would really spead
      // things up.
      // So this kernel does (2 * BLOCK_SIZE) iterations, which may seem a lot,
      // but we do at least do the I/O in an efficient way and keep the
      // inner loop simple and fast (e.g. no exp() or log()).
      int s = threadIdx.x,
          t = i - s;
      if (t >= 0) {
        // p_buf is indexed by s + 1 and t + 1 because it has an extra initial
        // row and column for context from previous blocks.  Taking into account
        // the way these buffers relate to the tensors p, px and py, and
        // ignoring `normalizer`, code below can be interpreted as follows,
        // writing sbb for s_block_begin and tbb for t_block_begin:
        //
        //   p[b][s+sbb][t+tbb] = LogAdd(p[b][s+sbb-1][t+tbb] + px[s+sbb-1][t+tbb],
        //                               p[b][s+sbb][t+tbb-1] + py[s+sbb][t+tbb-1]
        //
        // where you can see that apart from the offsets of tbb and sbb, this is
        // the same as the recursion defined for p in
        // mutual_information.py:mutual_information_recursion().
#if 1
        p_buf[s + 1][t + 1] = p_buf[s][t + 1] * px_buf[s][t] + p_buf[s + 1][t] * py_buf[s][t];
#else
        // This is an optimization of the statement above (the other half of
        // this #if/#else) where we keep p_buf[s + 1][t] in a register to avoid
        // the need for a load from shared memory.
        p_buf_s1_t = p_buf[s][t + 1] * px_buf[s][t] + p_buf_s1_t * py_buf[s][t];
        p_buf[s + 1][t + 1] = p_buf_s1_t;
#endif
      }
      __syncthreads();
    }

    // Write out the data.
    for (int i = threadDim.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int t = i % BLOCK_SIZE, s = i / BLOCK_SIZE;
      if (s < block_S && t < block_T) {
        float this_p = p_buf[s + 1][t + 1];
        p[b][s + s_block_begin][t + t_block_begin] = normalizer + log(this_p);
        if (this_p - this_p != 0 || this_p == 0)
          p_buf[0][0] = 1.0;  // This is a "panic" flag.
      }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      // Write `ans`, if this is the final (top-right) block in its sequence
      // Logically, the following equation corresponds to:
      //   ans[b] = p[b][s_end][t_end]
      if (s_block_begin + S > s_end && t_block_begin + T > t_end)
        ans[b] = normalizer + log(p_buf[s_end - s_block_begin + 1][t_end - t_block_begin + 1]);
    }



    if (p_buf[0][0] != 0.0) {
      // "panic" flag set.  We need to re-do the computation using log-add.
      // This time we won't use the buffers, we'll just load and save from main
      // memory.  This code should very rarely be reached; and anyway, caching
      // should help us quite a bit.
      for (int i = 0; i < 2 * BLOCK_SIZE; i++) {
        int block_s = threadIdx.x,
            block_t = i - block_s;
        if (static_cast<unsigned int>(t) < static_cast<unsigned int>(block_T) &&
            block_s < block_S) {
          int s = block_s + s_block_begin,
              t = block_t + t_block_begin;
          float p_s1 = (s == 0  ? -INFINITY : p[b][s - 1][t]),
              p_t1 = (t == 0 ? -INFINITY : p[b][s][t - 1]),
              this_px = px[b][s][t], this_py = py[b][s][t];
          float this_p = LogAdd(p_s1 + this_px,
                                p_t1 + this_py);
          if (i == 0 && is_origin_block)
            this_p = 0.0;
          p[b][s][t] = this_p;
        }
      }
      if (threadIdx.x == 0) {
        // Write `ans`, if this is the final (top-right) block in its sequence.
        // This is only reached in the 'panic situation' where we had overflow.
        if (s_block_begin + S > s_end && t_block_begin + T > t_end)
          ans[b] = p[b][s_end][t_end];
      }
    }
  }
}



/*
  Backward of mutual_information.

  If we were to write the forward pass in non-log space, it would be (ignoring
  edge cases), as follows... we'll prefix all the variable names with e, e.g. ep,
  to clarify that it's the exp of the actual argument p:

         ep[b][s][t] = ep[b][s - 1][t] * epx[b][s - 1][t] +
                       ep[b][s][t - 1] * epy[b][s][t - 1].    (eq. 1)

(A)
  First we consider the part that involves recursion, i.e. the part involving only gradients of
  ep.  The backprop involving ep only would be:
          ep_grad[b][s - 1][t] += ep_grad[b][s][t] * epx[b][s - 1][t]
          ep_grad[b][s][t - 1] += ep_grad[b][s][t] * epy[b][s][t - 1].
  .. and if we add 1 to the s index of the first equation above and 1 to the
     t index of the second equation, we can see that:

          ep_grad[b][s][t] = ep_grad[b][s + 1][t] * epx[b][s][t] +
                             ep_grad[b][s][t + 1] * epy[b][s][t].

  Now, if ep = exp(p),  then ep_grad == dy/dep == dy/dp dp/dep == dy/dp / (dep/dp) == dy/dp / exp(p)
                                     == dy/dp / ep.  == p_grad / ep.
                        I.e. ep_grad = p_grad / ep.
  So we can write the above as:
        p_grad[b][s][t] / ep[b][s][t] = p_grad[b][s + 1][t] / ep[b][s + 1][t] * epx[b][s][t] +
                                        p_grad[b][s][t + 1] / ep[b][s][t + 1] * epy[b][s][t].

  Or, rearranging:
       p_grad[b][s][t]  = p_grad[b][s + 1][t] * exp(p[b][s][t] + px[b][s][t] - p[b][s + 1][t]) +
                          p_grad[b][s][t + 1] * exp(p[b][s][t] + py[b][s][t] - p[b][s][t + 1]).   (eq. 2)

 (B)  The following is the backprop for epx and epy from (eq. 1):

        epx_grad[b][s - 1][t] +=  ep_grad[b][s][t] * ep[b][s - 1][t]
        epy_grad[b][s][t - 1] +=  ep_grad[b][s][t] * ep[b][s][t - 1]

   .. adding 1 to the s indexes in the 1st equation and to the t indexes in the 2nd:

        epx_grad[b][s][t] = ep_grad[b][s + 1][t] * ep[b][s][t]
        epy_grad[b][s][t] = ep_grad[b][s][t + 1] * ep[b][s][t]

    Using, similar to the above, ep_grad = p_grad / ep, and similarly,
    epx_grad = px_grad / epx and epy_grad = py_grad / epy, and writing exp(p) for p and so on,
    the above becomes

        px_grad[b][s][t] / exp(px[b][s][t]) =  p_grad[b][s + 1][t] / exp(p[b][s + 1][t] * exp(p[b][s][t])
        py_grad[b][s][t] / exp(py[b][s][t]) =  p_grad[b][s][t + 1] / exp(p[b][s][t + 1] * exp(p[b][s][t])
     Rearranging:
        px_grad[b][s][t]  = p_grad[b][s + 1][t] * exp(p[b][s][t] + px[b][s][t] - p[b][s + 1][t])  (eq. 3a)
        py_grad[b][s][t]  = p_grad[b][s][t + 1] * exp(p[b][s][t] + py[b][s][t] - p[b][s][t + 1])  (eq. 3b)


   Defining terms that are common to (eq. 2) and (eqs. 3a,3b), write:

      xderiv[b][s][t] := exp(p[b][s][t] + px[b][s][t] - p[b][s + 1][t])    (eq. 4)
      yderiv[b][s][t] := exp(p[b][s][t] + py[b][s][t] - p[b][s][t + 1])    (eq. 5)

   .. and note that these quantities are <= 1 so there is no problem doing
   the exponentiation.  So the recursion can be simplified as:

       p_grad[b][s][t]  = p_grad[b][s + 1][t] * xderiv[b][s][t] +
                          p_grad[b][s][t + 1] * yderiv[b][s][t]            (eq. 6)
       px_grad[b][s][t] = p_grad[b][s + 1][t] * yderiv[b][s][t]            (eq. 7)
       py_grad[b][s][t] = p_grad[b][s][t + 1] * yderiv[b][s][t]            (eq. 8)

  (It might seem like we could just reuse px_grad and py_grad for (eq. 6), but it's
  not clear to me that this is the best strategy since that would require an extra
  write to shared memory within the loop that's the limiting factor.)

  The backward pass will be slightly different from the forward pass in terms of
  how we store p (and p_grad), because for writing a particular block of p_grad, we
  need context on the top and right instead of the bottom and left.
 */
template <typename scalar_t>
__global__
void mutual_information_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3> px,   // B, S, T + 1, i.e. batch, x_seq_length, y_seq_length + 1
    torch::PackedTensorAccessor32<scalar_t, 3> py,   // B, S + 1, T.
    torch::PackedTensorAccessor32<scalar_t, 3> p,    // B, S + 1, T + 1.  Produced in forward pass.
    torch::PackedTensorAccessor32<scalar_t, 1> ans_grad,  // [B].  This is an input.
    torch::PackedTensorAccessor32<scalar_t, 3> p_grad,   // B, S + 1, T + 1.   This is a temporary.
    torch::PackedTensorAccessor32<scalar_t, 3> px_grad,   // B, S, T + 1.
    torch::PackedTensorAccessor32<scalar_t, 3> py_grad,   // B, S + 1, T.
    torch::PackedTensorAccessor32<int64_t, 2> boundary,  // B, 4;  or 0, 0 if boundaries are the defaults (0, 0, S, T)
    int iter) {    // This kernel is sequentially called with 'iter' = num_iters
                   // - 1, num_iters - 2, .. 0, where num_iters can be taken to
                   // be any sufficiently large number but will actually be:
                   // num_s_blocks + num_t_blocks - 1 where num_s_blocks = S /
                   // BLOCK_SIZE + 1 and num_t_blocks = T / BLOCK_SIZE + 1
  const int B = px.size(0),
      S = px.size(1),
      T = py.size(2);

  // For statements that are the same as the forward pass, we are omitting some comments
  // what we made there.  We'll focus, in the comments, on differences from the forward pass.
  const int num_s_blocks = S / BLOCK_SIZE + 1,
      num_t_blocks = T / BLOCK_SIZE + 1,
      num_blocks_this_iter = min(iter + 1, num_s_blocks);


  // px_buf and py_buf are used temporarily to store the px and py values,
  // but then modified to store the "xderiv" and "yderiv" values defined
  // in (eq. 5) and (eq. 6) above.  For out-of-range values, we'll write 0.0
  // here.
  //  px_buf[s][t] contains px[s+s_block_begin][t+t_block_begin];
  //  py_buf[s][t] contains py[s+s_block_begin][t+t_block_begin].
  // Unlike in the forward code, there is no offset of 1 in the indexes.
  __shared__ scalar_t px_buf[BLOCK_SIZE][BLOCK_SIZE],
      py_buf[BLOCK_SIZE][BLOCK_SIZE];


  // p_buf is initially used to store p, and then (after we are done putting
  // xderiv and yderiv into px_buf and py_buf) it is repurposed to store
  // p_grad.
  //
  // Unlike in the forward pass, p_buf has the same numbering as px_buf and
  // py_buf not offset by 1: e.g., for the origin block, p_buf[0][0] refers
  // to p[0][0] and not p[-1][-1].  The p_buf block is larger by 1 than
  // the block for px_buf and py_buf; unlike in the forward pass, we store
  // context on the top right, not the bottom left, i.e. the elements at
  // (one past the largest indexes in the block).
  //
  // For out-of-range elements of p_buf, we'll put zero.
  __shared__ scalar_t p_buf[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

  // boundary_buf will be used to store the b'th row of `boundary` if we have
  // boundary information supplied.
  __shared__ int64_t boundary_buf[4];

  if (threadIdx.x == 0) {
    boundary_buf[0] = 0;
    boundary_buf[1] = 0;
    boundary_buf[2] = S;
    boundary_buf[3] = T;
  }

  // batch_block_iter iterates over both batch elements (index b), and block
  // indexes in the range [0..num_blocks_this_iter-1].  The order here
  // doesn't matter, since there are no interdependencies between these
  // blocks (they are on a diagonal).
  for (int batch_block_iter = blockIdx.x;
       batch_block_iter < B * num_blocks_this_iter;
       batch_block_iter += gridDim.x) {
    int b = batch_block_iter % B,
        block = batch_block_iter / B;
    int s_block_begin = block * BLOCK_S_SIZE,
        t_block_begin = (iter  - block) * BLOCK_T_SIZE;

    if (threadDim.x < 4 && boundary.size(0) != 0)
      boundary_buf[threadDim.x] = boundary[b][threadDim.x];
    __syncthreads();

    int s_begin = boundary_buf[0],
        t_begin = boundary_buf[1],
        s_end = boundary_buf[2],
        t_end = boundary_buf[3];
    s_block_begin += s_begin;
    t_block_begin += t_begin;

    // block_S and block_T are the actual sizes of this block, no greater than
    // (BLOCK_SIZE, BLOCK_SIZE) but possibly less than that if we are towards
    // the end of the sequence.
    // The last element of the output matrix p we write is (s_end, t_end),
    // i.e. the one-past-the-end index is (s_end + 1, t_end + 1).
    int block_S = min(BLOCK_SIZE, s_end + 1 - s_block_begin),
       block_T = min(BLOCK_SIZE, t_end + 1 - t_block_begin);

    if (block_S <= 0 || block_T <= 0)
      continue;

    // Load px_buf and py_buf.  At this point they just contain px and py
    // for this block.
    for (int i = threadDim.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int s_in_block = i / BLOCK_SIZE,
          t_in_block = i % BLOCK_SIZE,
          s = s_in_block + s_block_begin,
          t = t_in_block + t_block_begin;
      // We let ps and py default to -infinity if they are out of range, which will
      // cause xderiv and yderiv for out-of-range values to be zero, and cause
      // correct behavior in edge cases (for the top and right blocks).
      // The issue is that p and p_grad are of larger size than px and py.
      scalar_t this_px = -INFINITY;
      if (s < s_end && t <= t_end)
        this_px = px[b][s - 1][t];
      px_buf[s_in_block][t_in_block] = this_px;
      scalar_t this_py = -INFINITY;
      if (s <= s_end && t < t_end)
        this_py = py[b][s][t - 1];
      py_buf[s_in_block][t_in_block] = this_py;
    }


    // load p.  This time we loop over the exact indexes we need.  Above
    // we looped to BLOCK_SIZE * BLOCK_SIZE rather than block_S and block_T
    // because having power-of-2 arrangement of threads may be helpful
    // for aligned reads, but here the loop is up to  (BLOCK_SIZE + 1) * (BLOCK_SIZE + 1)
    // which is not a power of 2, so that is not a concern here.
    for (int i = threadDim.x; i < (BLOCK_SIZE + 1) * (BLOCK_SIZE + 1); i += blockDim.x) {
      int s_in_block = i / (BLOCK_SIZE + 1),  // 0 <= s_in_block <= block_S
          t_in_block = i % (BLOCK_SIZE + 1),    // 0 <= t_in_block <= block_T
                   s = s_in_block + s_block_begin,
                   t = t_in_block + t_block_begin;
      // Setting 0.0 for out-of-bounds elements, together with setting
      // -INFINITY for out-of-bounds elements of px_buf and py_buf, will
      // ensure that we do the right thing in top and right edge cases,
      // i.e. that no derivatives will be propagated from out-of-bounds points.
      p_buf[s_in_block][t_in_block] = (s <= s_end && t <= t_end ?
                                       p[b][s][t] : 0.0);
    }

    // Set xderiv and yderiv; see (eq. 4) and (eq. 5).
    for (int i = threadDim.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      // We can apply this formula to the entire block even if we are processing
      // a partial block; elements outside the partial block will not be used so
      // their values don't matter, and elements just out
      int t = i % BLOCK_SIZE, s = i / BLOCK_SIZE;
      // Mathematically the following is doing:
      //   xderiv[b][s][t] := exp(p[b][s][t] + px[b][s][t] - p[b][s + 1][t])
      // (with an offset on the s and t indexes)
      px_buf[s][t] = exp(px_buf[s][t] + px_buf[s][t] - p_buf[s + 1][t]);
      // Mathematically the following is doing:
      //   yderiv[b][s][t] := exp(p[b][s][t] + py[b][s][t] - p[b][s][t + 1])
      // (with an offset on the s and t indexes)
      py_buf[s][t] = exp(px_buf[s][t] + py_buf[s][t] - p_buf[s][t + 1]);
    }

    // Load p_grad for the top and right elements in p_buf: i.e. for elements
    // p_buf[s][t] where s == block_S (exclusive-or) t == block_T.  We don't
    // need to load the top-right corner [block_S][block_T]; that location will
    // never be accessed.
    // These are the p_grad values computed by previous instances of this kernel
    // If this is one of the top or right blocks, some or all of the p_grad
    // values we'd be reading here will be out of range, and we use zeros.
    if (threadIdx.x < block_S) {
      int s_in_block = threadIdx.x,
          t_in_block = block_T,
          s = s_in_block + s_block_begin,
          t = t_in_block + t_block_begin;
      p_buf[s_in_block][t_in_block] = (
          s <= s_end && t <= t_end ? p_grad[s][t] : 0.0);
    } else if (static_cast<unsigned int>(threadIdx.x - 64) <
               static_cast<unsigned int>(block_T)) {
      int s_in_block = block_S,
          t_in_block = threadIdx.x - 64,
                   s = s_in_block + s_block_begin,
                   t = t_in_block + t_block_begin;
      p_buf[s_in_block][t_in_block] = (
          s <= s_end && t <= t_end ? p_grad[s][t] : 0.0);
    }

    // The number of inner iterations, i.e. iterations inside this
    // kernel, is this_num_inner_iters.  The highest iteration,
    // corresponding to the highest-indexed value of p_buf that
    // we need to set,
    //  corresponds to p_buf[block_S - 1][block_T - 1],
    // and the iteration number is the sum of these indexes, i.e.
    // (block_S - 1) + (block_T - 1).

    bool is_final_block = (s_block_begin + block_S == s_end + 1 &&
                           t_block_begin + block_T == t_end + 1);

    int first_iter = block_S + block_T - 2;
    if (is_final_block) {
      // The following statement, mathematically, corresponds to:
      // p_grad[b][s_end][t_end] = ans_grad[b] Normally this element of p_buf
      // would be set by the first iteration of the loop below, so if it's set
      // this way we have to decrement first_iter to prevent it being
      // overwritten.
      p_buf[block_S - 1][block_T - 1] = ans_grad[b];
      --first_iter;
    }

    for (int i = first_iter; i >= 0; --i) {
      int s = i,
          t = i - threadIdx.x;
      if (t >= 0) {
        // The following statement is really operating on the gradients;
        // it corresponds to (eq. 6) defined above, i.e.:
        //   p_grad[b][s][t]  = p_grad[b][s + 1][t] * xderiv[b][s][t] +
        //                      p_grad[b][s][t + 1] * yderiv[b][s][t]
        p_buf[s][t] = (p_buf[s + 1][t] * px_buf[s][t] +
                       p_buf[s][t + 1] * py_buf[s][t]);
      }
    }

    // Write out p_grad, px_grad and py_grad.
    for (int i = threadDim.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int t_in_block = i % BLOCK_SIZE,
         s_in_block = i / BLOCK_SIZE,
         s = s_in_block + s_block_begin,
         t = t_in_block + t_block_begin;
      if (t <= t_end && s <= s_end) {
        p_grad[b][s][t] = p_buf[s_in_block][t_in_block];

        if (s < s_end) {  // write px_grad, which is of shape [B][S][T + 1]
          // From (eq. 7):
          // px_grad[b][s][t] = p_grad[b][s + 1][t] * yderiv[b][s][t]
          px_grad[b][s][t] = (p_buf[s_in_block + 1][t_in_block] *
                              px_buf[s_in_block][t_in_block]);
        }
        if (t < t_end) {  // write py_grad, which is of shape [B][S + 1][T]
          // from (eq. 8):
          // py_grad[b][s][t] = p_grad[b][s][t + 1] * yderiv[b][s][t]
          py_grad[b][s][t] = (p_buf[s_in_block][t_in_block + 1] *
                              py_buf[s_in_block][t_in_block]);
        }
      }
    }

    if (threadIdx.x == 0 && s_block_begin == s_begin &&
        t_block_end == t_end)
      ans_grad[b] = p_buf[0][0];
  }
}




// forward of mutual_information.  See """... """ comment of `mutual_information` in
// mutual_information.py for documentation of the behavior of this function.
torch::Tensor mutual_information_cuda(torch::Tensor px,
                                      torch::Tensor py,
                                      std::optional<torch::Tensor> optional_boundary,
                                      torch::Tensor p) {
  TORCH_CHECK(px.dim() == 3, "px must be 3-dimensional");
  TORCH_CHECK(py.dim() == 3, "py must be 3-dimensional.");
  TORCH_CHECK(p.dim() == 3, "p must be 3-dimensional.");
  TORCH_CHECK(px.device().is_cuda() && py.device().is_cuda() && p.device().is_cuda(),
              "inputs must be CUDA tensors");

  auto scalar_t = px.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(px.device());

  const int B = px.size(0),
      S = px.size(1),
      T = px.size(2) - 1;
  TORCH_CHECK(py.size(0) == B && py.size(1) == S + 1 && py.size(2) == T);
  TORCH_CHECK(p.size(0) == B && p.size(1) == S + 1 && p.size(2) == T + 1);

  torch::Tensor ans = torch::empty({B}, opts);

  int num_threads = 128,
      num_blocks = 128,
      BLOCK_SIZE = 32;

  const int num_s_blocks = S / BLOCK_SIZE + 1,
      num_t_blocks = T / BLOCK_SIZE + 1,
      num_iters = num_s_blocks + num_t_blocks - 1;

  bool has_boundary = (bool)optional_boundary;
  if (!has_boundary)
    optional_boundary = torch::empty({0, 0}, long_opts);

  for (int iter = 0; iter < num_iters; ++iter) {
    mutual_information_kernel<scalar_t, BLOCK_SIZE><<<num_blocks, num_threads>>>(
        px.packed_accessor32<scalar_t, 3>(),
        py.packed_accessor32<scalar_t, 3>(),
        p.packed_accessor32<scalar_t, 3>(),
        optional_boundary.value().packed_accessor32<int64_t, 2>(),
        ans.packed_accessor32<scalar_t, 1>(),
        iter);
  }
  return ans;
}



// backward of mutual_information; returns (grad_px, grad_py)
torch::Tensor mutual_information_backward_cuda(torch::Tensor px,
                                               torch::Tensor py,
                                               std::optional<torch::Tensor> optional_boundary,
                                               torch::Tensor p,
                                               torch::Tensor ans_grad) {
  TORCH_CHECK(px.dim() == 3, "px must be 3-dimensional");
  TORCH_CHECK(py.dim() == 3, "py must be 3-dimensional.");
  TORCH_CHECK(p.dim() == 3, "p must be 3-dimensional.");
  TORCH_CHECK(ans_grad.dim() == 1, "ans_grad must be 1-dimensional.");


  TORCH_CHECK(px.device().is_cuda() && py.device().is_cuda() &&
              p.device().is_cuda() && ans_grad.device().is_cuda() &&
              "inputs must be CUDA tensors");

  auto scalar_t = px.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_t).device(px.device());

  const int B = px.size(0),
      S = px.size(1),
      T = px.size(2) - 1;

  TORCH_CHECK(py.size(0) == B && py.size(1) == S + 1 && py.size(2) == T);
  TORCH_CHECK(p.size(0) == B && p.size(1) == S + 1 && p.size(2) == T + 1);

  torch::Tensor p_grad = torch::empty({B, S + 1, T + 1}, opts),
      px_grad = torch::empty({B, S, T + 1}, opts),
      py_grad = torch::empty({B, S + 1, T}, opts),

  const int num_threads = 128,
      num_blocks = 128,
      BLOCK_SIZE = 32;

  const int num_s_blocks = S / BLOCK_SIZE + 1,
      num_t_blocks = T / BLOCK_SIZE + 1,
      num_iters = num_s_blocks + num_t_blocks - 1;

  bool has_boundary = (bool)optional_boundary;
  if (!has_boundary)
    optional_boundary = torch::empty({0, 0}, long_opts);

  for (int iter = num_iters - 1; iter >= 0; --iter) {
    mutual_information_backward_kernel<scalar_t, BLOCK_SIZE><<<num_blocks, num_threads>>>(
        px.packed_accessor32<scalar_t, 3>(),
        py.packed_accessor32<scalar_t, 3>(),
        p.packed_accessor32<scalar_t, 3>(),
        ans_grad.packed_accessor32<scalar_t, 1>,
        p_grad.packed_accessor32<scalar_t, 3>(),
        px_grad.packed_accessor32<scalar_t, 3>(),
        py_grad.packed_accessor32<scalar_t, 3>(),
        optional_boundary.value().packed_accessor32<int64_t, 2>(),
        iter);
  }
  return std::vector<torch::Tensor>({px_grad, py_grad});
}
