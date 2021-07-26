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



def _mutual_information_forward_dispatcher(input: torch.Tensor,
                                       params: torch.Tensor) -> torch.Tensor:
    if input.is_cuda:
        if torch_mutual_information_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_mutual_information_cuda.mutual_information_cuda(
            input, params.contiguous())
    else:
        return torch_mutual_information_cpu.mutual_information_cpu(
            input, params)

def _mutual_information_backward_dispatcher(input: torch.Tensor,
                                        params: torch.Tensor,
                                        grad_output) -> Tuple[torch.Tensor, torch.Tensor]:
    if input.is_cuda:
        if torch_mutual_information_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return tuple(torch_mutual_information_cuda.mutual_information_backward_cuda(
            input, params,
            grad_output))
    else:
        return tuple(torch_mutual_information_cpu.mutual_information_backward_cpu(
            input, params, grad_output))



def _reshape_as_3dim(x: torch.Tensor, dim: int):
    """
    Returns x reshaped so that dimension 'dim' is the middle of 3 dimensions,
    combining dimensions and unsqueezing as needed.  For example (writing
    the behavior of this function as
        input_shape, dim -> output_shape,
    it will do:
              (3), 0 -> (1, 3, 1)
        (2, 5, 9), 1 -> (2, 5, 9)
        (2, 5, 9), 2 -> (10, 9, 1)
        (3, 4, 5, 6) -> (12, 5, 6)
    The idea is to normalize the shape so the channel dimension is the middle
    of 3, so the implementation can deal with a fixed layout.

     Args:
        x:  tensor to be reshaped
      dim:  Dimension of x that is to be the middle of 3 dimensions in the result.
            If negative, interpreted as an offset from x.dim.
    """
    if dim < 0:
        dim += input.ndim
    orig_shape = list(x.shape)

    # `new_shape` is `orig_shape` but modified so that the channel dim (`dim`)
    # is dimension/axis 1.  We do this not by transposing, but by combining
    # adjacent dims.
    a, b = 1, 1
    for i in range(0, dim):
        a *= orig_shape[i]
    for i in range(dim + 1, len(orig_shape)):
        b *= orig_shape[i]
    new_shape = (a, orig_shape[dim], b)
    return x.reshape(new_shape)  # `reshape` will make a contiguous copy if needed.


class LearnedNonlinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, params: torch.Tensor, dim: int) -> torch.Tensor:
        if dim < 0:
            dim += input.ndim
        assert dim >= 0 and dim < input.ndim
        assert params.ndim == 2 and params.shape[1] % 2 == 1
        assert params.shape[0] == input.shape[dim]

        ctx.dim = dim
        ctx.save_for_backward(input, params)
        output = _mutual_information_forward_dispatcher(_reshape_as_3dim(input, dim),
                                                    params)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, None) -> Tuple[torch.Tensor, torch.Tensor, None]:
        (input, params) = ctx.saved_tensors
        orig_shape = input.shape
        # We re-do the reshaping in the backward, rather than save the reshaped
        # input, so that if this reshaping results in a copy it is not retained
        # (this saves memory at the expense of a little extra work in such
        # situations).
        grad_input, grad_params = _mutual_information_backward_dispatcher(
            _reshape_as_3dim(input, ctx.dim), params, grad_output)
        return grad_input.reshape(input.shape), grad_params, None


def mutual_information(input, params, dim):
    """Learned nonlinearity.
    Args:
       input:   The input, to be transformed pointwise; may be of any shape.

       params:  The parameters of the learned nonlinearity.  Interpreted
                as of shape (C, N + 1), where C is the channel and N, which
                must be a power of 2 more than 1, is the number of linear regions in the
                piecewise linear function.  The first element is the log
                of the distance between the discontinuities, and the
                remaining elements are the derivatives of the function
                in the linear pieces.  We can explain what this function
                is as follows:
                     Let the row of `params` for a particular channel be
                interpreted as (l, d0, d1, d2 ... ).  Let K = N/2, and L = exp(l).
                Then the discontinuities in the function are at:
                    L * ( -K+1, -K+2, .., -1, 0, 1, .. K-1 )
                and the values d0, d1 .. are interpreted as the slopes of the
                function in the intervals, respectively:
                    [-inf.. L*(-K+1)), [L*-K+1..L*-K+2], ...
                and we use these together with the assumption that the
                function's value at x=0 is 0, to compute the function's value.

                In terms of concrete calculations, we do it as follows:
                Firstly, we can get rid of the factor of L by treating the l
                parameter as a scale on the input and output, i.e.:
                  x = input * exp(-l)
                ... do the calculation y = f(xwithout a scale, interpreting the
                discontinuities as being at integer values -K+1, -K+2, ... K+1,
                and then:
                  output = y * = output * exp(l)

                The core computation requires computing the y-values at the
                discontinuities at -K+1, -K+2 and so on.  Each one equals
                the sign of the offset (- for negative K) times the sum
                of the derivatives 'd' for the regions between the current
                points and zero.  If we number these as offsets o0, o1 and
                so on up to N-2, then the formula is:

                     for o_n with n < K, o_N = -sum(k = n+1..K-1) d_k
                     for o_n with n >= k, o_N = sum(K..n-1) d_k

                  e.g. if K=3 and  (d0, d1, d2, d3, d4, d5) = (1, 2, 1, 2, 1, 1), then:

                  o_0 = -(d1+d2) = -3    # x=-2 maps to y=-3
                  o_1 = -(d2) = -2       # x=-1 maps to y=-2
                  o_2 = () = 0           # x=0 maps to y=0
                  o_3 = (d3) = 2         # x=1 maps to y=2
                  o_4 = (d3 + d4) = 3    # x=2 maps to y=3

        dim:  The dimension of `input` that corresponds to the channel.  It is
              recommended that the channel should not be the fastest-varying
              dimension (the one with stride=1), because this will make
              the data loads and stores be non-coalesced and the kernels
              will be quite slow.

          Return: output, of the same shape as `input`.
    """
    return LearnedNonlinFunction.apply(x, params, dim)
