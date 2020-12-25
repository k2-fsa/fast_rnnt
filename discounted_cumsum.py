import time

import torch
from torch.utils.cpp_extension import load


torch_discounted_cumsum = load(
    name='torch_discounted_cumsum',
    sources=['discounted_cumsum.cpp', 'discounted_cumsum_kernel.cu'],
    verbose=True,
)


# class DiscountedCumSumFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weights, bias, old_h, old_cell):
#         outputs = torch_discounted_cumsum.forward(input, weights, bias, old_h, old_cell)
#         new_h, new_cell = outputs[:2]
#         variables = outputs[1:] + [weights]
#         ctx.save_for_backward(*variables)
#
#         return new_h, new_cell
#
#     @staticmethod
#     def backward(ctx, grad_h, grad_cell):
#         outputs = torch_discounted_cumsum.backward(
#             grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
#         d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
#         return d_input, d_weights, d_bias, d_old_h, d_old_cell


def discounted_cumsum_right_minthreads(input, gamma):
    return torch_discounted_cumsum.discounted_cumsum_right_minthreads(input, gamma)


def discounted_cumsum_right_coalesced(input, gamma):
    return torch_discounted_cumsum.discounted_cumsum_right_coalesced(input, gamma)


def discounted_cumsum_right_gold(input, gamma):
    assert input.dim() == 2
    assert 0 <= gamma <= 1
    out = []
    last_col = torch.zeros((input.shape[0], 1), dtype=input.dtype, device=input.device)
    for i in reversed(range(input.shape[1])):
        cur_col = input[:, i].unsqueeze(-1)
        last_col = cur_col + gamma * last_col
        out.insert(0, last_col)
    out = torch.cat(out, dim=1)
    return out


def test_fn(fn):
    torch.manual_seed(0)
    x = torch.full((10, 10000), fill_value=1.0, dtype=torch.float32).cuda()
    gamma = 0.99
    out_gold_32 = discounted_cumsum_right_gold(x, gamma)
    out_gold_64 = discounted_cumsum_right_gold(x.double(), gamma)
    out_fn = fn(x, gamma)
    diff_32 = (out_fn - out_gold_32).abs().max().item()
    diff_64 = (out_fn - out_gold_64).abs().max().item()
    print(fn.__name__)
    print('diff_32', diff_32)
    print('diff_64', diff_64)


def test_speed(fn, reps=10000):
    torch.manual_seed(0)
    x = torch.randn(10, 100000, dtype=torch.float32).cuda()
    gamma = 0.99
    t1 = time.time()
    for _ in range(reps):
        fn(x, gamma)
    t2 = time.time()
    print(fn.__name__, t2-t1)


if __name__ == '__main__':
    test_fn(discounted_cumsum_right_minthreads)
    test_fn(discounted_cumsum_right_coalesced)
    test_speed(discounted_cumsum_right_minthreads)
    test_speed(discounted_cumsum_right_coalesced)
