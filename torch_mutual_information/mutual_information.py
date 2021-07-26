import os

import torch
from typing import Tuple
from torch.utils.cpp_extension import load

VERBOSE = False


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
                                           boundaries: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if input.is_cuda:
        if torch_mutual_information_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_mutual_information_cuda.mutual_information_cuda(
            px, py, boundaries, q)
    else:
        return torch_mutual_information_cpu.mutual_information_cpu(
            px, py, boundaries, q)

def _mutual_information_backward_dispatcher(px: torch.Tensor, py: torch.Tensor,
                                            boundaries: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if px.is_cuda:
        if torch_mutual_information_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return tuple(torch_mutual_information_cuda.mutual_information_backward_cuda(
            px, py, boundaries, q))

    else:
        return tuple(torch_mutual_information_cpu.mutual_information_backward_cpu(
            px, py, boundaries, q))



class MutualInformationRecursionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, px: torch.Tensor, py: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        (B, S, T) = px.shape

        # q is a rearrangement of a tensor p which is of shape (B,S,T),
        # using p[b,s,t] == q[b,s+t,t].  The reason for working with this
        # representation is that each row of q depends only on the previous row,
        # so we can access the rows sequenctially and this leads to
        # better memory access patterns.  We are assuming that most likely
        # T < S, which means that q should not require much more memory than p.
        #
        # Actually we access q beginning from 0 indexes even if `boundaries`
        # has t_begin > 0 or s_begin > 0, i.e. we really access q as
        #   q[b, s-s_begin + t-t_begin, t-t_begin];
        # note, rows of `boundaries` are [s_begin, t_begin, s_end, t_end].
        # We don't need q if we are not going to do backprop

        q = (torch.empty(B, S + T, device=px.device, dtype=px.dtype)
             if px.requires_grad or py.requires_grad
             else None)

        ans = _mutual_information_forward_dispatcher(px, py, boundaries, q)

        if px.requires_grad or py.requires_grad:
            ctx.save_for_backward(px, py, boundaries, w)

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        (px, py, boundaries, q) = ctx.saved_tensors
        (px_grad, py_grad) = _mutual_information_backward_dispatcher(px, py, boundaries, q)
        return (px_grad, py_grad, None)



def mutual_information_recursion(input, px, py, boundaries=None):
    """A recursion that is useful in computing mutual information between two sequences of
    real vectors, but may be useful more generally in sequence-to-sequence tasks where
    monotonic alignment between pairs of sequences is desired.  The definitions of
    the arguments are definitions that would be used when computing this type of
    mutual information, but you can also view them as arbitrary quantities and just
    look at the formula computed by this function.

    Args:
          px:  A torch.Tensor of some floating point type, with shape [B][S][T],
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
          py:  A torch.Tensor of the same dtype as px, with shape [B][S][T],
               representing
                 py[b][s][t] =  log [ p(y_t | x_{0..s-1}, y_{0..t-1}) / p(y_t) ]
               This function does not treat x and y differently; the only difference
               is that the implementation assumes for optimization purposes that y
               is likely to be the shorter sequence, i.e. that "most of the time T < S",
               and it will be faster if you respect this.
          boundaries:  If supplied, a torch.LongTensor of shape [B][4], where each row contains
               [s_begin, t_begin, s_end, t_end].  If not supplied, the values
               [0, 0, S, T] will be assumed.  These are the beginning and
               one-past-the-last positions in the x and y sequences
               respectively, and can be used if not all sequences are of the same length.

    Returns:
        Returns a torch.Tensor of shape [B], containing the log of the mutuafl
        information between the b'th pair of sequences.  This is defined by
        the following recursion on p[b,s,t] (where p is of shape [B,S,T]),
        representing a mutual information between sub-sequences of lengths s and t:

             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s,t], p[b,s,t-1] + py[b,s,t])

        where in the case where boundaries==None: the edge cases are handled
        by treating p[b,-1,-1] as 0 and all other quantities with negative
        indexes as -infinity; and ans[b] would equal p[S-1,T-1].  The extension to
        cases where the boundaries are specified should be obvious.

    """
    assert px.ndim == 3 and px.shape == py.shape and px.dtype == py.dtype
    (B, S, T) = px.shape
    if boundaries is not None:
        assert boundaries.dtype == torch.LongTensor
        assert boundaries.shape == (B, 4)

    return MutualInformationRecursion.apply(px, py, boundaries)
