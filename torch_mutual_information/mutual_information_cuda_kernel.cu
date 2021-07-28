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
    int iter) {    // This kernel is sequentially called with 'iter' = 0, 1, 2 and so on, up to:
                   //    (S+BLOCK_S_SIZE-1)/BLOCK_S_SIZE + (T+BLOCK_T_SIZE-1)/BLOCK_T_SIZE  - 1
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

    bool is_origin_block = (s_block_begin * t_block_begin == 0);

    int s_end, t_end;  // s_end and t_end are the end points (last-plus-one) of the entire sequence.
    if (threadDim.x < 4 && boundary.size(0) != 0)
      boundary_buf[threadDim.x] = boundary[b][threadDim.x];
    __syncthreads();
    int s_begin = boundary_buf[0],
        t_begin = boundary_buf[1];
    s_end = boundary_buf[2];
    t_end = boundary_buf[3];
    s_block_begin += s_begin;
    t_block_begin += t_begin;

    // block_S and block_T are the actual sizes of this block, no greater than
    // (BLOCK_SIZE, BLOCK_SIZE) but possibly less than that if we are towards
    // the end of the sequence.
    int block_S = min(BLOCK_SIZE, s_end - s_block_begin),
        block_T = min(BLOCK_SIZE, t_end - t_block_begin);

    if (block_S <= 0 || block_T <= 0)
      continue;


    // Load px_buf and py_buf.  We exponentiate; the assumption is that they most likely
    // won't overflow or underflow, but if they do overflow we'll detect it later; we'll
    // also detect certain kinds of underflow.
    for (int i = threadDim.x; i < BLOCK_SIZE * BLOCK_SIZE; i += blockDim.x) {
      int t_in_block = i % BLOCK_SIZE,
          s_in_block = i / BLOCK_SIZE,
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

    for (int i = 1; i < 2 * BLOCK_SIZE; i++) {
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
       px_grad[b][s][t] = p_grad[b][s][t + 1] * yderiv[b][s][t]            (eq. 7)
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
    int iter) {    // This kernel is sequentially called with 'iter' = num_iters - 1, num_iters - 2, .. 0,
                   // where num_iters can be taken to be any sufficiently large number but will actually be:
                   //    (S+BLOCK_S_SIZE-1)/BLOCK_S_SIZE + (T+BLOCK_T_SIZE-1)/BLOCK_T_SIZE  - 1
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

  boundary_buf[0] = 0;
  boundary_buf[1] = 0;
  boundary_buf[2] = S;
  boundary_buf[3] = T;


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
      *params_buf = (scalar_t*) y_vals + 3 + N;  // [N].  Contains parameters (not times scale!)
                                                 // Caution: contains params[c][1] through params[c][N],
                                                 // i.e. numbering is off by 1 versus params.
                                                 //  params_buf[-1] contains params[c][0] == log of scale;
                                                 // params_buf[-2] and params_buf[-3] contain scale and inv_scale.

  __shared__ scalar_t input_buf[THREADS_PER_BLOCK];  // input sequence
  __shared__ scalar_t output_grad_buf[THREADS_PER_BLOCK];
  __shared__ char n_buf[THREADS_PER_BLOCK];  // for each input in `input_buf`,
                                             // this stores the integer value 0
                                             // <= n < N which determines which
                                             // piece of the piecewise linear
                                             // function we are in.

  // Load parameters
  if (threadIdx.x <= N)
    params_buf[threadIdx.x - 1] = params[c][threadIdx.x];
  __syncthreads();

  if (threadIdx.x == 0) {
    scalar_t scale = exp(params_buf[-1]);
    params_buf[-2] = scale;
    params_buf[-3] = 1.0 / scale;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    scalar_t scale = params_buf[-2],
        sum_positive = 0.0;
    for (int i = 0; i < K; i++) {
      // params_buf is indexed with an index one less than params.
      scalar_t pos_scaled_param = params_buf[K + i] * scale;
      y_vals[K + i] = sum_positive - pos_scaled_param * i;
      sum_positive += pos_scaled_param;
    }
  } else if (threadIdx.x == 64) {
    scalar_t scale = params_buf[-2],
        sum_negative = 0.0;
    for (int i = 0; i < K; i++) {
      scalar_t neg_scaled_param = params_buf[K - i - 1] * scale;
      sum_negative -= neg_scaled_param;
      y_vals[K - i - 1] = sum_negative + neg_scaled_param * (i + 1);
    }
  }
  __syncthreads();


  // this_param_grad and this_y_grad pertain to the 'n' value (i.e. the n'th
  // linear interval) corresponding to n == threadIdx.x % N.  For example, if
  // threadIdx.x == 0, this thread's gradient corresponds to the left-most
  // linear interval.
  scalar_t this_param_grad = 0.0,
      this_y_vals_grad = 0.0;

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
      // The following is equivalent to:
      // int t = (threadIdx.x % T_inc) + t_offset;
      // given that T_inc is a power of 2 and t_offset >= THREADS_PER_BLOCK >= T_inc.
      int t = (threadIdx.x & (T_inc - 1)) | t_offset;

      scalar_t this_input = 0.0, this_output_grad;
      if (t < T) {
        this_output_grad = output_grad[b][c][t];
        this_input = input[b][c][t];
        input_buf[threadIdx.x] = this_input;
        output_grad_buf[threadIdx.x] = this_output_grad;
      }
      scalar_t x = this_input * inv_scale + K;
      if (x < 0) x = 0;
      else if (x >= N) x = N - 1;

      // The forward code did:
      // output[b][c][t] = this_input * params_buf[n] + y_vals[n];
      // We get the derivative for params and y_vals later.
      if (t < T) {
        int n = (int)x;   // C++ rounds toward zero.
        n_buf[threadIdx.x] = (char)n;
        input_grad[b][c][t] = this_output_grad * params_buf[n];
      } else {
        n_buf[threadIdx.x] = 255;
      }

      int this_block_start = threadIdx.x & ~(N-1),  // == N * (threadIdx.x / N),
                                                    // since N is power of 2
          this_n = threadIdx.x & (N-1); // == threadIdx.x % N.
      // this_n is the n value that this thread accumulates gradients for;
      // it is responsible for output_grads in the block of threads
      // from this_block_start to this_block_start+N-1.


      // __syncthreads();  // <- not really needed.
      // At this point there is an implicit within-warp
      // synchronization (Note: implicit warp synchronization is not considered
      // future-proof).  Threads above have written to n_buf, and threads below
      // will read from it; but we don't need to explicitly synchronize for now
      // because the reads/writes are among threads in a group of N threads with
      // (4 <= N <= 16); and 16 is less than the warp size which is 32 or 64.

      // src_indexes will contain up to 16 16-bit numbers, stored starting in its
      // least significant bits.  It will store all the offsets within this
      // block of N threads, whose chosen 'n' value equals this_n.
      uint64_t src_indexes = 0;
      // num_src is the number of numbers in `src_indexes`.  We need to store a
      // separate counter because zero is a valid index and if we are to support
      // N == 16 we don't have bits to spare in src_indexes to store some kind
      // of marker.
      int num_src = 0;

      // This loop always does at least N statements, but they should be
      // relatively fast ones since the computation per n value is minimal and
      // there is little I/O.  We are figuring out the subset of our block of N
      // elements, which this particular thread value is responsible for
      // (because they have n == this_n), and storing them in `src_indexes` and
      // `num_src`.
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
      for (; num_src > 0; --num_src, (src_indexes >>= 4)) {
        int src_thread = this_block_start | (src_indexes & 0xF);
        scalar_t src_output_grad = output_grad_buf[src_thread],
            src_input = input_buf[src_thread];
        assert(n_buf[src_thread] == this_n);
        n_buf[src_thread] = 0;
        // Backprop for: output = input * params_buf[n] + y_vals[n].
        // Here, n == this_n; this is how we selected these `src_idx` values.
        this_param_grad += src_output_grad * src_input;
        this_y_vals_grad += src_output_grad;
      }

      // TODO: remove the next lines
      assert(n_buf[threadIdx.x] == 0 || (unsigned char)n_buf[threadIdx.x] == 255);
      output_grad_buf[threadIdx.x] = 0.0;
    }
  }

  __syncthreads();  // sync threads because we are about to re-use
                    // output_grad_buf for reduction, and, later, input_buf.

  this_param_grad = strided_reduce_sum(N, output_grad_buf, this_param_grad);
  __syncthreads();
  this_y_vals_grad = strided_reduce_sum(N, output_grad_buf, this_y_vals_grad);

  __syncthreads();  // sync threads because we are about to re-use
                    // output_grad_buf as y_vals_grad_buf.

  // Re-use some buffers..
  scalar_t *params_grad_buf = input_buf + 1,  // [N]  ... but element [-1] will have deriv of scale.
      *y_vals_grad_buf = output_grad_buf;   // [N]

  if (threadIdx.x < N) {
    params_grad_buf[threadIdx.x] = this_param_grad;
    y_vals_grad_buf[threadIdx.x] = this_y_vals_grad;
  }
  __syncthreads(); // other threads are about to read params_grad_buf and
                   // y_vals_grad_buf.

  // This next block does backprop relating to `y_vals`.  Comparing with the CPU
  // version (call this the "reference code") is the best way to understand this
  // (this code is just a modification of that).  The main difference is we
  // modify the indexes into params and params_grad by -1, so the index
  // corresponds to the 'n' value; and element -1 of params_grad_buf will have
  // the deriv of the log scale.

  scalar_t l_grad;
  if (threadIdx.x == 0) {
    // Now do the backprop for the loop above where we set y_vals_a.  This could
    // be further optimized to replace the loop with a raking, but I doubt this
    // will have a huge effect on the runtime since K will be fairly small,
    // e.g. 4.
    scalar_t scale = params_buf[-2],
        scale_grad = 0.0,
        sum_positive_grad = 0.0;
    for (int i = K - 1; i >= 0; i--) {
      // Backprop for: sum_positive += pos_scaled_param;
      scalar_t pos_scaled_param_grad = sum_positive_grad;
      // Backprop for: y_vals[K + i] = sum_positive - pos_scaled_param * i;
      scalar_t y_grad_pos = y_vals_grad_buf[K + i];
      pos_scaled_param_grad -= i * y_grad_pos;
      sum_positive_grad += y_grad_pos;
      // Backprop for: pos_scaled_param = params_buf[K + i] * scale,
      params_grad_buf[K + i] += pos_scaled_param_grad * scale;
      scale_grad += pos_scaled_param_grad * params_buf[K + i];
    }
    // Backprop for: scale = exp(l), where l = params[c][0].
    l_grad = scale * scale_grad;
  } else if (threadIdx.x == 64) {
    // Now do the backprop for the loop above where we set y_vals.
    // Make this one threadIdx.x == 0 so it's possibly quicker to test
    //
    scalar_t scale = params_buf[-2],
        scale_grad = 0.0,
        sum_negative_grad = 0.0;
    for (int i = K - 1; i >= 0; i--) {
      // Backprop for: y_vals[K - i - 1] = sum_negative + neg_scaled_param * (i + 1):
      scalar_t y_grad_neg = y_vals_grad_buf[K - i - 1];
      sum_negative_grad += y_grad_neg;
      scalar_t neg_scaled_param_grad = y_grad_neg * (i + 1);
      // Backprop for: sum_negative -= neg_scaled_param;
      neg_scaled_param_grad -= sum_negative_grad;
      // Backprop for: neg_scaled_param = params_buf[K - i - 1] * scale;
      params_grad_buf[K - i - 1] += neg_scaled_param_grad * scale;
      scale_grad += neg_scaled_param_grad * params_buf[K - i - 1];
    }
    params_grad_buf[-1] = scale * scale_grad;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    params_grad_buf[-1] += l_grad;  // contribution to l grad from the "negative" branch
  }
  __syncthreads();
  if (threadIdx.x <= N) {
    params_grad[blockIdx.y][c][threadIdx.x] = params_grad_buf[threadIdx.x - 1];
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
      num_blocks = 128;

  const int num_s_blocks = S / BLOCK_SIZE + 1,
      num_t_blocks = T / BLOCK_SIZE + 1,
      num_iters = std::max<int>(num_s_blocks, num_t_blocks);

  bool has_boundary = (bool)optional_boundary;
  if (!has_boundary)
    optional_boundary = torch::empty({0, 0}, long_opts);

  for (int iter = 0; iter < num_iters; iter++) {
    mutual_information_kernel<scalar_t, 32><<<num_blocks, num_threads>>>(
        px.packed_accessor32<scalar_t, 3>(),
        py.packed_accessor32<scalar_t, 3>(),
        p.packed_accessor32<scalar_t, 3>(),
        optional_boundary.value().packed_accessor32<int64_t, 2>(),
        ans.packed_accessor32<scalar_t, 1>(),
        iter);
  }




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
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mutual_information_kernel", ([&] {
        mutual_information_kernel<scalar_t><<<gridDim, THREADS_PER_BLOCK, sizeof(scalar_t) * shared_mem_numel, at::cuda::getCurrentCUDAStream()>>>(
              input.packed_accessor32<scalar_t, 3>(),
              params.packed_accessor32<scalar_t, 2>(),
              output.packed_accessor32<scalar_t, 3>(),
              images_per_thread_block);
      }));
  return output;
}



std::vector<torch::Tensor> mutual_information_backward_cuda(torch::Tensor input,
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
  TORCH_CHECK((N & (N-1)) == 0, "N must be a power of 2")

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

  TORCH_CHECK(THREADS_PER_BLOCK / images_per_thread_block >= N);

  torch::Tensor params_grad = torch::zeros({grid_dim_y, C, N + 1}, opts);

  dim3 gridDim(C, grid_dim_y, 1);

  // blockDim is scalar, just THREADS_PER_BLOCK.
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mutual_information_backward_kernel", ([&] {
        mutual_information_backward_kernel<scalar_t><<<gridDim, THREADS_PER_BLOCK, sizeof(scalar_t) * shared_mem_numel, at::cuda::getCurrentCUDAStream()>>>(
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
