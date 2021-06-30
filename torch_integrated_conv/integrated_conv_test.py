import torch
from torch_integrated_conv import integrated_conv


def test_integrated_conv_zeros():
    N = 1
    C = 2
    H = 3
    W = 4
    for device in [ torch.device('cpu'), torch.device('cuda:0') ]:
        for dtype in [torch.float32, torch.float64]:
            print("device=", device, ", dtype=", dtype)
            input = torch.zeros(N, 2 * C, H, W, device=device, dtype=dtype)
            kH = 5
            kW = 5
            pos_add = torch.zeros(C, kH, kW, device=device, dtype=dtype)
            pos_mul = torch.zeros(C, kH, kW, device=device, dtype=dtype)

            output = integrated_conv(input, pos_add, pos_mul)
            assert torch.allclose(input, output)
