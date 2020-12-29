import torch
from torch.utils.cpp_extension import load


torch_discounted_cumsum_cpu = load(
    name='torch_discounted_cumsum_cpu',
    sources=['discounted_cumsum_cpu.cpp'],
    # verbose=True,
)

torch_discounted_cumsum_cuda = None
if torch.cuda.is_available():
    torch_discounted_cumsum_cuda = load(
        name='torch_discounted_cumsum_cuda',
        sources=['discounted_cumsum_cuda.cpp', 'discounted_cumsum_cuda_kernel.cu'],
        verbose=True,
    )


def _discounted_cumsum_left_dispatcher(input, gamma):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.discounted_cumsum_left_cuda(input.contiguous(), gamma)
    else:
        return torch_discounted_cumsum_cpu.discounted_cumsum_left_cpu(input, gamma)


def _discounted_cumsum_right_dispatcher(input, gamma):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.discounted_cumsum_right_cuda(input.contiguous(), gamma)
    else:
        return torch_discounted_cumsum_cpu.discounted_cumsum_right_cpu(input, gamma)


class DiscountedCumSumLeftFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        output = _discounted_cumsum_left_dispatcher(input, gamma)
        ctx.save_for_backward(torch.tensor(gamma))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        gamma = ctx.saved_variables[0].item()
        grad_input = _discounted_cumsum_right_dispatcher(grad_output, gamma)
        return grad_input, None


class DiscountedCumSumRightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        output = _discounted_cumsum_right_dispatcher(input, gamma)
        ctx.save_for_backward(torch.tensor(gamma))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        gamma = ctx.saved_variables[0].item()
        grad_input = _discounted_cumsum_left_dispatcher(grad_output, gamma)
        return grad_input, None


def discounted_cumsum_left(input, gamma):
    return DiscountedCumSumLeftFunction.apply(input, gamma)


def discounted_cumsum_right(input, gamma):
    return DiscountedCumSumRightFunction.apply(input, gamma)
