import os
import random
import time
import unittest

import torch
from tqdm import tqdm

from torch_discounted_cumsum import discounted_cumsum_left, discounted_cumsum_right


def get_grad(param, out):
    out.sum().backward()
    grad = param.grad.clone()
    del param.grad
    return grad


def discounted_cumsum_left_gold(input, gamma):
    assert input.dim() == 2
    assert 0 <= gamma <= 1
    out = []
    last_col = torch.zeros((input.shape[0], 1), dtype=input.dtype, device=input.device)
    for i in range(input.shape[1]):
        cur_col = input[:, i].unsqueeze(-1)
        last_col = cur_col + gamma * last_col
        out.append(last_col)
    out = torch.cat(out, dim=1)
    return out


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


def discounted_cumsum_lib(x, gamma, dir):
    return {
        'left': discounted_cumsum_left,
        'right': discounted_cumsum_right,
    }[dir](x, gamma)


def discounted_cumsum_gold(x, gamma, dir):
    return {
        'left': discounted_cumsum_left_gold,
        'right': discounted_cumsum_right_gold,
    }[dir](x, gamma)


def compute_linf(batchsz, veclen, dir, gamma=0.99, dtype=torch.float32, cuda=False, data='randn', tol=1e-3, seed=2020):
    torch.manual_seed(seed)
    if data == 'randn':
        x = torch.randn((batchsz, veclen), dtype=dtype)
    elif data == 'ones':
        x = torch.ones((batchsz, veclen), dtype=dtype)
    else:
        raise ValueError('Invalid data generation identifier')
    if cuda:
        x = x.cuda()
    x = torch.nn.Parameter(x)

    out_gold = discounted_cumsum_gold(x, gamma, dir)
    grad_gold = get_grad(x, out_gold)

    out_lib = discounted_cumsum_lib(x, gamma, dir)
    grad_lib = get_grad(x, out_lib)

    out_linf = (out_lib - out_gold).abs().max().item()
    grad_linf = (grad_lib - grad_gold).abs().max().item()

    if out_linf >= tol or grad_linf >= tol:
        print(f'x={x}\nout_gold={out_gold}\nout_lib={out_lib}\ngrad_gold={grad_gold}\ngrad_lib={grad_lib}\n')

    return out_linf, grad_linf


class TestDiscountedCumSum(unittest.TestCase):
    def test_validity(self):
        print('Testing validity...')
        is_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''
        for cuda in (True, False):
            if cuda and not is_cuda:
                print('Skipping validity CUDA tests')
                continue
            rng = random.Random(2020)
            with tqdm(total=2*2*2*17) as pbar:
                for data in ('ones', 'randn'):
                    for dtype in (torch.float32, torch.float64):
                        for i in range(2):
                            batchsz = 8 ** i
                            for j in range(17):
                                veclen = max(1, 2 ** j + rng.randint(-1, 1))
                                gamma = rng.random()
                                seed = rng.randint(0, 2 ** 16)
                                dir = rng.choice(['left', 'right'])
                                tol = 2e-3
                                out_linf, grad_linf = compute_linf(
                                    batchsz, veclen, dir, gamma, dtype, cuda, data, tol, seed
                                )
                                msg = f'Validity test failed with batchsz={batchsz}, veclen={veclen}, dir={dir}, ' \
                                      f'gamma={gamma}, dtype={dtype}, cuda={cuda}, data={data}, seed={seed}, ' \
                                      f'out_linf={out_linf}, grad_linf={grad_linf}'
                                self.assertLess(out_linf, tol, msg)
                                self.assertLess(grad_linf, tol, msg)
                                pbar.update(1)

    def test_precision(self):
        print('Testing precision...')
        is_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''
        if not is_cuda:
            print('Skipping precision tests')
            return
        batchsz = 1
        veclen = 10000
        gamma = 0.99
        dir = 'right'
        for data in ('ones', 'randn'):
            if data == 'ones':
                precision_factor = 2.0
            else:
                precision_factor = 1.1
            torch.manual_seed(2020)
            if data == 'randn':
                x_32 = torch.randn((batchsz, veclen), dtype=torch.float32)
            elif data == 'ones':
                x_32 = torch.ones((batchsz, veclen), dtype=torch.float32)
            else:
                raise ValueError('Invalid data generation identifier')
            x_32 = x_32.cuda()
            x_64 = x_32.double()

            gold_64 = discounted_cumsum_gold(x_64, gamma, dir)
            gold_32 = discounted_cumsum_gold(x_32, gamma, dir).double()
            lib_32 = discounted_cumsum_lib(x_32, gamma, dir).double()

            err_32_gold = (gold_32 - gold_64).abs().max().item()
            err_32_lib = (lib_32 - gold_64).abs().max().item()

            msg = f'Precision improvement test failed with data={data}, ' \
                  f'err_32_gold={err_32_gold}, err_32_lib={err_32_lib}'
            self.assertLess(precision_factor * err_32_lib, err_32_gold, msg)

            print(f'data={data}\nerr_32_gold={err_32_gold:10.8f}\nerr_32_lib ={err_32_lib:10.8f}')

    def test_speed(self):
        print('Testing speed...')
        is_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''
        NUM_RUNS = 30
        NUM_RUNS_GOLD = 6
        if not is_cuda:
            print('Skipping speed tests')
            return
        gamma = 0.99
        x_32 = torch.randn((1, 100000), dtype=torch.float32)
        x_32 += torch.ones_like(x_32)
        x_32_gpu = x_32.cuda()

        timer = time.clock_gettime(time.CLOCK_MONOTONIC)
        for _ in tqdm(range(NUM_RUNS_GOLD), desc='gold', leave=True):
            discounted_cumsum_right_gold(x_32, gamma)
        dur_gold = time.clock_gettime(time.CLOCK_MONOTONIC) - timer
        dur_gold = dur_gold * NUM_RUNS / NUM_RUNS_GOLD

        timer = time.clock_gettime(time.CLOCK_MONOTONIC)
        for _ in tqdm(range(NUM_RUNS), desc='lib_cpu', leave=True):
            discounted_cumsum_right(x_32, gamma)
        dur_lib_cpu = time.clock_gettime(time.CLOCK_MONOTONIC) - timer

        timer = time.clock_gettime(time.CLOCK_MONOTONIC)
        for _ in tqdm(range(NUM_RUNS), desc='lib_cuda', leave=True):
            discounted_cumsum_right(x_32_gpu, gamma)
        dur_lib_cuda = time.clock_gettime(time.CLOCK_MONOTONIC) - timer

        print(f'dur_gold: {dur_gold:7.4f} sec')
        print(f'dur_lib_cpu: {dur_lib_cpu:7.4f} sec')
        print(f'dur_lib_cuda: {dur_lib_cuda:7.4f} sec')
        print(f'speedup gold -> lib_cpu: {dur_gold / dur_lib_cpu:5.2f}')
        print(f'speedup gold -> lib_cuda: {dur_gold / dur_lib_cuda:5.2f}')
        print(f'speedup lib_cpu -> lib_cuda: {dur_lib_cpu / dur_lib_cuda:5.2f}')


if __name__ == '__main__':
    unittest.main()
