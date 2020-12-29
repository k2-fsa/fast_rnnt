import time
import torch

from discounted_cumsum import discounted_cumsum_left, discounted_cumsum_right


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


def test_left():
    torch.manual_seed(0)
    x = torch.full((10, 10000), fill_value=1.0, dtype=torch.float32).cuda()
    gamma = 0.99
    out_gold_32 = discounted_cumsum_left_gold(x, gamma)
    out_gold_64 = discounted_cumsum_left_gold(x.double(), gamma)
    out_fn = discounted_cumsum_left(x, gamma)
    diff_32 = (out_fn - out_gold_32).abs().max().item()
    diff_64 = (out_fn - out_gold_64).abs().max().item()
    print('left diff_32', diff_32)
    print('left diff_64', diff_64)


def test_right():
    torch.manual_seed(0)
    x = torch.full((10, 10000), fill_value=1.0, dtype=torch.float32).cuda()
    gamma = 0.99
    out_gold_32 = discounted_cumsum_right_gold(x, gamma)
    out_gold_64 = discounted_cumsum_right_gold(x.double(), gamma)
    out_fn = discounted_cumsum_right(x, gamma)
    diff_32 = (out_fn - out_gold_32).abs().max().item()
    diff_64 = (out_fn - out_gold_64).abs().max().item()
    print('right diff_32', diff_32)
    print('right diff_64', diff_64)


def test_grad_left():
    torch.manual_seed(0)
    x = torch.full((10, 10000), fill_value=1.0, dtype=torch.float32).cuda()
    x = torch.nn.Parameter(x)
    gamma = 0.99

    out_gold = discounted_cumsum_left_gold(x, gamma)
    out_gold.sum().backward()
    out_gold_grad = x.grad.clone()

    del x.grad

    out_fn = discounted_cumsum_left(x, gamma)
    out_fn.sum().backward()
    out_fn_grad = x.grad.clone()

    diff_grad = (out_gold_grad - out_fn_grad).abs().max().item()
    print('left diff_grad', diff_grad)


def test_grad_right():
    torch.manual_seed(0)
    x = torch.full((10, 10000), fill_value=1.0, dtype=torch.float32).cuda()
    x = torch.nn.Parameter(x)
    gamma = 0.99

    out_gold = discounted_cumsum_right_gold(x, gamma)
    out_gold.sum().backward()
    out_gold_grad = x.grad.clone()

    del x.grad

    out_fn = discounted_cumsum_right(x, gamma)
    out_fn.sum().backward()
    out_fn_grad = x.grad.clone()

    diff_grad = (out_gold_grad - out_fn_grad).abs().max().item()
    print('right diff_grad', diff_grad)


def test_speed(reps=10000):
    torch.manual_seed(0)
    x = torch.randn(10, 100000, dtype=torch.float32).cuda()
    gamma = 0.99
    t1 = time.time()
    for _ in range(reps):
        discounted_cumsum_right(x, gamma)
    t2 = time.time()
    print('sec:', t2-t1)


if __name__ == '__main__':
    test_left()
    test_right()
    test_grad_left()
    test_grad_right()
    #test_speed()
