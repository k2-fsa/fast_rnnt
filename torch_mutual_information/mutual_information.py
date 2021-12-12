import os

import torch
from torch import Tensor
from typing import Tuple, Optional, Sequence
from torch.utils.cpp_extension import load

VERBOSE = False
# DEBUG = False

def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    import torch_mutual_information_cpu
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_mutual_information_cpu')
    torch_mutual_information_cpu = load(
        name='torch_mutual_information_cpu',
        sources=[
            _resolve('mutual_information_cpu.cpp'),
        ],
        verbose=VERBOSE,
    )


try:
        import torch_mutual_information_cuda
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_mutual_information_cuda')
    torch_mutual_information_cuda = None
    if torch.cuda.is_available():
        torch_mutual_information_cuda = load(
            name='torch_mutual_information_cuda',
            sources=[
                _resolve('mutual_information_cuda.cpp'),
                _resolve('mutual_information_cuda_kernel.cu'),
            ],
            verbose=VERBOSE,
        )



def _mutual_information_forward_dispatcher(px: torch.Tensor, py: torch.Tensor,
                                           boundary: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if px.is_cuda:
        if torch_mutual_information_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_mutual_information_cuda.mutual_information_cuda(
            px, py, boundary, p)
    else:
        return torch_mutual_information_cpu.mutual_information_cpu(
            px, py, boundary, p)

def _mutual_information_backward_dispatcher(px: torch.Tensor, py: torch.Tensor,
                                            boundary: torch.Tensor, p: torch.Tensor,
                                            ans_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if px.is_cuda:
        if torch_mutual_information_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        overwrite_ans_grad = True
        if overwrite_ans_grad:
            ans_grad_copy = ans_grad.clone()
        ans = tuple(torch_mutual_information_cuda.mutual_information_backward_cuda(
            px, py, boundary, p, ans_grad_copy, overwrite_ans_grad))
        if overwrite_ans_grad:
            if not torch.allclose(ans_grad, ans_grad_copy, rtol=1.0e-02):
                print(f"Warning: possible excesssive roundoff in mutual information backward "
                      f"recursion: {ans_grad} vs. {ans_grad_copy}");
        return ans
    else:
        return tuple(torch_mutual_information_cpu.mutual_information_backward_cpu(
            px, py, boundary, p, ans_grad))



class MutualInformationRecursionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, px: torch.Tensor, py: torch.Tensor, boundary: Optional[torch.Tensor]) -> torch.Tensor:
        (B, S, T1) = px.shape
        T = T1 - 1;
        assert py.shape == (B, S + 1, T)
        if boundary is not None:
            assert boundary.shape == (B, 4)
        else:
            boundary = torch.zeros(0, 0, dtype=torch.int64, device=px.device)


        # p is a tensor of shape (B, S + 1, T + 1) were p[s][t] is the
        # the mutual information of the pair of subsequences of x and y that are of
        # length s and t respectively.  p[0][0] will be 0.0 and p[S][T] is
        # the mutual information of the entire pair of sequences, i.e. of lengths
        # S and T respectively.
        # It is computed as follows (in C++ and CUDA):
        #       p[b,0,0] = 0.0
        #       p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
        #                          p[b,s,t-1] + py[b,s,t-1])
        #               if s > 0 or t > 0,
        #               treating values with any -1 index as -infinity.
        #      .. if `boundary` is set, we start fom p[b,s_begin,t_begin]=0.0.

        p = torch.empty(B, S + 1, T + 1, device=px.device, dtype=px.dtype)

        ans = _mutual_information_forward_dispatcher(px, py, boundary, p)

        #if DEBUG:
        #    print("p = ", p)

        # print(f"p = {p}, boundary = {boundary}, psum={p.sum()}")

        if px.requires_grad or py.requires_grad:
            ctx.save_for_backward(px, py, boundary, p)
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        (px, py, boundary, p) = ctx.saved_tensors
        (px_grad, py_grad) = _mutual_information_backward_dispatcher(px, py, boundary, p, ans_grad)
        return (px_grad, py_grad, None)



def mutual_information_recursion(px, py, boundary=None):
    """A recursion that is useful in computing mutual information between two sequences of
    real vectors, but may be useful more generally in sequence-to-sequence tasks where
    monotonic alignment between pairs of sequences is desired.  The definitions of
    the arguments are definitions that would be used when computing this type of
    mutual information, but you can also view them as arbitrary quantities and just
    make use of the formula computed by this function.

    Args:
          px:  A torch.Tensor of some floating point type, with shape [B][S][T+1],
               where B is the batch size, S is the length of the 'x' sequence
               (including representations of EOS symbols but not BOS symbols), and S is the
               length of the 'y' sequence (including representations of
               EOS symbols but not BOS symbols).  In the mutual information application,
               px[b][s][t] would represent the following log odds ratio; ignoring
               the b index on the right to make the notation more compact,

                 px[b][s][t] =  log [ p(x_s | x_{0..s-1}, y_{0..t-1}) / p(x_s) ]

               This expression also implicitly includes the log-probability of
               choosing to generate an x value as opposed to a y value.  In
               practice it might be computed as a + b, where a is the log
               probability of choosing to extend the sequence of length (s,t)
               with an x as opposed to a y value; and b might in practice be
               of the form:
                   log(N exp f(x_s, y_{t-1}) / sum_t'  exp f(x_s, y_t'))
               where N is the number of terms that the sum over t' included, which
               might include some or all of the other sequences as well as this one.

               Note: we don't require px and py to be contiguous, but the
               code assumes for optimization purposes that the T axis has
               stride 1.

          py:  A torch.Tensor of the same dtype as px, with shape [B][S+1][T],
               representing
                 py[b][s][t] =  log [ p(y_t | x_{0..s-1}, y_{0..t-1}) / p(y_t) ]
               This function does not treat x and y differently; the only difference
               is that for optimization purposes we assume the last axis (the t axis)
               has stride of 1; this is true if px and py are contiguous.

          boundary:  If supplied, a torch.LongTensor of shape [B][4], where each row contains
               [s_begin, t_begin, s_end, t_end], with 0 <= s_begin <= s_end < S and
               0 <= t_begin <= t_end < T (this implies that empty sequences are allowed).  If not supplied, the values
               [0, 0, S, T] will be assumed.  These are the beginning and
               one-past-the-last positions in the x and y sequences
               respectively, and can be used if not all sequences are of the same length.

    Returns:
        Returns a torch.Tensor of shape [B], containing the log of the mutual
        information between the b'th pair of sequences.  This is defined by
        the following recursion on p[b,s,t] (where p is of shape [B,S+1,T+1]),
        representing a mutual information between sub-sequences of lengths s and t:

             p[b,0,0] = 0.0
             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                p[b,s,t-1] + py[b,s,t-1])
                       (if s > 0 or t > 0)

        where we handle edge cases by treating quantities with negative indexes
        as -infinity.  The extension to cases where the boundaries are specified
        should be obvious; it just works on shorter sequences with offsets into
        px and py.
    """
    assert px.ndim == 3
    B, S, T1 = px.shape
    T = T1 - 1
    assert py.shape == (B, S + 1, T)
    assert px.dtype == py.dtype
    (B, S, T) = px.shape
    if boundary is not None:
        assert boundary.dtype == torch.int64
        assert boundary.shape == (B, 4)
        for [ s_begin, t_begin, s_end, t_end ] in boundary.to('cpu').tolist():
            assert 0 <= s_begin <= s_end <= S
            assert 0 <= t_begin <= t_end <= T
    # The following assertions are for efficiency
    assert px.stride()[-1] == 1
    assert py.stride()[-1] == 1
    return MutualInformationRecursionFunction.apply(px, py, boundary)


def _inner(a: Tensor, b: Tensor) -> Tensor:
    """
    Does inner product on the last dimension, with expected broadcasting, i.e. equivalent to
     (a * b).sum(dim=-1)
    without creating a large temporary.
    """
    assert a.shape[-1] == b.shape[-1]  # last last dim be K
    a = a.unsqueeze(-2)  # (..., 1, K)
    b = b.unsqueeze(-1) # (..., K, 1)
    c = torch.matmul(a, b)  # (..., 1, 1)
    return c.squeeze(-1).squeeze(-1)


def joint_mutual_information_recursion(px: Sequence[Tensor], py: Sequence[Tensor],
                                       boundary: Optional[Tensor] = None) -> Sequence[Tensor]:
    """A recursion that is useful for modifications of RNN-T and similar loss functions,
    where the recursion probabilities have a number of terms and you want them reported
    separately.  See mutual_information_recursion() for more documentation of the
    basic aspects of this.

    Args:
      px: a sequence of Tensors, each of the same shape [B][S][T+1]
      py: a sequence of Tensor, each of the same shape [B][S+1][T], the sequence must be
              the same length as px.
      boundary: optionally, a LongTensor of shape [B][4] containing rows
               [s_begin, t_begin, s_end, t_end], with 0 <= s_begin <= s_end < S and
               0 <= t_begin <= t_end < T, defaulting to [0, 0, S, T].
               These are the beginning and
               one-past-the-last positions in the x and y sequences
               respectively, and can be used if not all sequences are of the same length.
    Returns:
      a Tensor of shape (len(px), B),
      whose sum over dim 0 is the total log-prob of the recursion mentioned below, per sequence.
      The first element of the sequence of length len(px) is "special", in that it has an offset term
      reflecting the difference between sum-of-log and log-of-sum; for more interpretable
      loss values, the "main" part of your loss function should be first.

      The recursion below applies if boundary == None, when it defaults
      to (0, 0, S, T); where px_sum, py_sum are the sums of the elements of px and py:

          p = tensor of shape (B, S+1, T+1), containing -infinity
          p[b,0,0] = 0.0
          # do the following in loop over s and t:
          p[b,s,t] = log_add(p[b,s-1,t] + px_sum[b,s-1,t],
                              p[b,s,t-1] + py_sum[b,s,t-1])
                      (if s > 0 or t > 0)
          return b[:][S][T]

    This function lets you implement the above recursion efficiently, except
    that it gives you a breakdown of the contribution from all the elements of
    px and py separately.  As noted above, the first element of the
    sequence is "special".
    """
    N = len(px)
    assert len(py) == N and N > 0
    B, S, T1 = px[0].shape
    T = T1 - 1
    assert py[0].shape == (B, S + 1, T)
    assert px[0].dtype == py[0].dtype

    px_cat = torch.stack(px, dim=0)  # (N, B, S, T+1)
    py_cat = torch.stack(py, dim=0)  # (N, B, S+1, T)
    px_tot = px_cat.sum(dim=0)  # (B, S, T+1)
    py_tot = py_cat.sum(dim=0)  # (B, S+1, T)

    if boundary is not None:
        assert boundary.dtype == torch.int64
        assert boundary.shape == (B, 4)
        for [ s_begin, t_begin, s_end, t_end ] in boundary.to('cpu').tolist():
            assert 0 <= s_begin <= s_end <= S
            assert 0 <= t_begin <= t_end <= T
    else:
        boundary = torch.zeros(0, 0, dtype=torch.int64, device=px_tot.device)

    px_tot, py_tot = px_tot.contiguous(), py_tot.contiguous()
    # The following assertions are for efficiency
    assert px_tot.stride()[-1] == 1 and px_tot.ndim == 3
    assert py_tot.stride()[-1] == 1 and py_tot.ndim == 3

    p = torch.empty(B, S + 1, T + 1, device=px_tot.device, dtype=px_tot.dtype)

    # note, tot_probs is without grad.
    tot_probs = _mutual_information_forward_dispatcher(px_tot, py_tot, boundary, p)

    # this is a kind of "fake gradient" that we use, in effect to compute
    # occupation probabilities.  The backprop will work regardless of the
    # actual derivative w.r.t. the total probs.
    ans_grad = torch.ones(B, device=px_tot.device, dtype=px_tot.dtype)

    (px_grad, py_grad) = _mutual_information_backward_dispatcher(px_tot, py_tot,
                                                                 boundary, p, ans_grad)

    px_grad, py_grad = px_grad.reshape(1, B, -1), py_grad.reshape(1, B, -1)
    px_cat, py_cat = px_cat.reshape(N, B, -1), py_cat.reshape(N, B, -1)
    # get rid of -inf, would generate nan on product with 0
    px_cat, py_cat = px_cat.clamp(min=-1.0e+38), py_cat.clamp(min=-1.0e+38)

    x_prods = _inner(px_grad, px_cat) # (N, B)
    y_prods = _inner(py_grad, py_cat) # (N, B)


    # If all the occupation counts were exactly 1.0 (i.e. no partial counts),
    # "prods" should be equal to "tot_probs"; however, in general, "tot_probs"
    # will be more positive due to the difference between log-of-sum and
    # sum-of-log
    prods = x_prods + y_prods  # (N, B)
    with torch.no_grad():
        offset = tot_probs - prods.sum(dim=0)  # (B,)
    prods[0] += offset
    return prods # (N, B)
