import os

import torch
from typing import Tuple
from torch.utils.cpp_extension import load

VERBOSE = False


def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    import torch_integrated_conv_cpu
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_integrated_conv_cpu')
    torch_integrated_conv_cpu = load(
        name='torch_integrated_conv_cpu',
        sources=[
            _resolve('integrated_conv_cpu.cpp'),
        ],
        verbose=VERBOSE,
    )


try:
        import torch_integrated_conv_cuda
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_integrated_conv_cuda')
    torch_integrated_conv_cuda = None
    if torch.cuda.is_available():
        torch_integrated_conv_cuda = load(
            name='torch_integrated_conv_cuda',
            sources=[
                _resolve('integrated_conv_cuda.cpp'),
                _resolve('integrated_conv_cuda_kernel.cu'),
            ],
            verbose=VERBOSE,
        )



def _integrated_conv_forward_dispatcher(input: torch.Tensor,
                                       pos_add: torch.Tensor,
                                       pos_mul: torch.Tensor) -> torch.Tensor:
    if input.is_cuda:
        if torch_integrated_conv_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_integrated_conv_cuda.integrated_conv_cuda(
            input.contiguous(), pos_add.contiguous(), pos_mul.contiguous())
    else:
        return torch_integrated_conv_cpu.integrated_conv_cpu(
            input, pos_add, pos_mul)

def _integrated_conv_backward_dispatcher(input: torch.Tensor,
                                         pos_add: torch.Tensor,
                                         pos_mul: torch.Tensor,
                                         grad_output) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input.is_cuda:
        if torch_integrated_conv_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return tuple(torch_integrated_conv_cuda.integrated_conv_backward_cuda(
            input.contiguous(), pos_add.contiguous(), pos_mul.contiguous()))
    else:
        return tuple(torch_integrated_conv_cpu.integrated_conv_backward_cpu(
            input, pos_add, pos_mul))



class IntegratedConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, pos_add: torch.Tensor, pos_mul: torch.Tensor) -> torch.Tensor:
        output = _integrated_conv_forward_dispatcher(input, pos_add, pos_mul)
        ctx.save_for_backward(input, pos_add, pos_mul)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (input, pos_add, pos_mul) = ctx.saved_tensors
        grad_input, grad_pos_add, grad_pos_mul = _integrated_conv_backward_dispatcher(
            input, pos_add, pos_mul, grad_output)
        return grad_input, grad_pos_add, grad_pos_mul


def integrated_conv(input, pos_add, pos_mul):
    """Integrated convolution.
    Args:
       input:   The input of shape (N, 2*C, W) for 1-d convolution or (N, 2*C, H, W)
               for 2-d convolution, where
               N is the batch size, C is the number of output channels, and H and W are
               the input image's height and width respectively.  The input channels are
               of two types, "src" and "dest" respectively, meaning whether they relate
               to the source or destination image position; all the "src" channels come
               first, then the "dest" channels.
       pos_add:  Positional encoding: the additive part of the convolution kernel.
               This is of shape (C, kW) for 1-d
               convolution or (C, kH, kW) for 2-d convolution,
               where C is the number of channels and kH and kW are the kernel height and
               kernel width.  Kernel height and width must be odd (we assume zero padding
               so the output size is the same as the input size).
       pos_mul:  Positional encoding: the multiplicative part of the convolution kernel.
               This is of shape (C, kW)
               for 1-d convolution or (C, kH, kW) for 2-d convolution, where C
               is the number of channels and kH and kW are the kernel height and
               kernel width.
    Return: output, of shape (N, C, W) for 1-d convolution or (N, C, H, W) for
               2-d convolution.  In the 2-d case the output will be satisfy:

              output[n, c, h, w] = \sum_{kh=0}^{kH-1} \sum_{kw=0}^{kW-1}
                pos_mul[c, kh, kw] * relu(input[n, c, h, w] + input_padded[n,c,h+kh,w+kw] + pos_add[c, kh, kw])

              where input_padded is torch.pad(input, (kW//2, kW//2, kH//2, kH//2)),
              meaning zero-padding (this is done implicitly by the implementation).
              (Technically this is more closely related to cross-correlation than to
              convolution).
    """
    if input.ndim == 3:
        assert pos_add.ndim == 2 and pos_mul.ndim == 2
        # For now we choose to handle only the 2-dimensional case directly.  The
        # 1-dimensional one is treated as a special case of the 2-dimensional one.
        # Actually we could unsqueeze with -2 or -1 here, as the height and width
        # behave the same.
        return integrated_conv(input.unsqueeze(-2),
                               pos_add.unsqueeze(-2), pos_mul.unsqueeze(-2)).squeeze(-2)
    assert input.ndim == 4 and pos_add.ndim == 3 and pos_mul.ndim == 3
    assert input.shape[1] // 2 == pos_add.shape[0] == pos_mul.shape[0]
    return IntegratedConvFunction.apply(input, pos_add, pos_mul)
