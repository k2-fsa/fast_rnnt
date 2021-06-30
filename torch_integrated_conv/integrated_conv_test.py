import random
import torch
from torch_integrated_conv import integrated_conv


def test_integrated_conv_zeros():
    N = 1
    C = 2
    H = 3
    W = 4
    for device in [ torch.device('cpu'), torch.device('cuda:0') ]:
        if device == torch.device('cuda:0') and not torch.cuda.is_available():
            print("Warning: torch not available, not testing this part.")
            continue
        for dtype in [torch.float32, torch.float64]:
            print("device=", device, ", dtype=", dtype)
            input = torch.zeros(N, 2 * C, H, W, device=device, dtype=dtype)
            kH = 5
            kW = 5
            pos_add = torch.zeros(C, kH, kW, device=device, dtype=dtype)
            pos_mul = torch.zeros(C, kH, kW, device=device, dtype=dtype)

            output_ref = torch.zeros(N, C, H, W, device=device, dtype=dtype)
            output = integrated_conv(input, pos_add, pos_mul)
            assert torch.allclose(output, output_ref)


def test_integrated_conv_compare():
    N = 1
    C = 2
    H = 3
    W = 4
    if not torch.cuda.is_available():
        print("Warning: torch not available, not testing this part.")
        return
    for dtype in [torch.float32, torch.float64]:
        print("dtype=", dtype)
        input = torch.randn(N, 2 * C, H, W, dtype=dtype)
        device = torch.device('cuda:0')
        input_cuda = input.to(device)

        kH = 5
        kW = 5
        pos_add = torch.randn(C, kH, kW, dtype=dtype)
        pos_mul = torch.randn(C, kH, kW, dtype=dtype)

        pos_add_cuda = pos_add.to(device)
        pos_mul_cuda = pos_mul.to(device)

        output = integrated_conv(input, pos_add, pos_mul)
        output_cuda = integrated_conv(input_cuda, pos_add_cuda, pos_mul_cuda)
        print("output = ", output)
        print("output_cuda = ", output_cuda)
        diff = (output - output_cuda.to(torch.device('cpu'))).abs().sum()
        abs = output.abs().sum()
        print("Diff = ", diff, ", abs = ", abs)
        assert torch.allclose(output, output_cuda.to(torch.device('cpu')),
                              atol=1.0e-05)


def test_integrated_conv_rand_compare():
    for _ in range(30):
        N = random.randint(1, 256)
        C = random.randint(1, 64)
        H = random.randint(1, 128)
        W = random.randint(1, 128)

        while N * C * H * W > 65535:
            if N >= C and N >= H and N >= W:
                N = N // 2
            elif C >= H and C >= W:
                C = C // 2
            elif H >= W:
                H = H // 2
            else:
                W = W // 2


        if not torch.cuda.is_available():
            print("Warning: torch not available, not testing this part.")
            return
        for dtype in [torch.float32, torch.float64]:
            print("dtype=", dtype)
            input = torch.randn(N, 2 * C, H, W, dtype=dtype)
            device = torch.device('cuda:0')
            input_cuda = input.to(device)

            kH = random.randint(1, 10)
            kW = random.randint(1, 10)
            if kH % 2 == 0:
                kH += 1
            if kW % 2 == 0:
                kW += 1
            pos_add = torch.randn(C, kH, kW, dtype=dtype)
            pos_mul = torch.randn(C, kH, kW, dtype=dtype)
            pos_add_cuda = pos_add.to(device)
            pos_mul_cuda = pos_mul.to(device)

            output = integrated_conv(input, pos_add, pos_mul)
            output_cuda = integrated_conv(input_cuda, pos_add_cuda, pos_mul_cuda)

            diff = (output - output_cuda.to(torch.device('cpu'))).abs().sum()
            abs = output.abs().sum()
            print("Diff = ", diff, ", abs = ", abs)

            if not torch.allclose(output, output_cuda.to(torch.device('cpu')),
                                  atol=1.0e-05):
                print("output = ", output)
                print("output_cuda = ", output_cuda)
                assert 0, "outputs differ"
