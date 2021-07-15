import random
import torch
from torch_learned_nonlin import learned_nonlin


def test_learned_nonlin_basic():
    for dtype in [torch.float32, torch.float64]:
        B = 2
        C = 4
        T = 10
        x = -2.0 + 0.4 * torch.arange(10, dtype=dtype)
        x = x.reshape(1, 1, 10).repeat(B, C, 1)

        K = 4
        N = K * 2
        params = torch.arange(N + 1, dtype=dtype).unsqueeze(0) + torch.arange(C, dtype=dtype).unsqueeze(1)
        x.requires_grad = True
        params.requires_grad = True
        print("x = ", x)
        print("params = ", params)
        print("x.shape = ", x.shape)
        y = learned_nonlin(x, params, dim = 1)

        print("y = ", y)

        if torch.cuda.is_available():
            # test that the CUDA forward is the same as the CPU forward.
            device = torch.device('cuda:0')
            y2 = learned_nonlin(x.to(device), params.to(device), dim = 1).to(torch.device('cpu'))
            print("Checking CUDA is same")
            if not torch.allclose(y, y2, atol=1.0e-06):
                print(f"Error: CPU versus CUDA not the same: {y} vs. {y2}, diff = {y2-y}")
                assert(0);

        y.sum().backward()

        print("x.grad = ", x.grad)
        print("params.grad = ", params.grad)

        # Just eyeballing the above tgo make sure it looks reasonable.


def test_learned_nonlin_deriv():
    """ Tests derivatives in randomized way """
    for _ in range(10):
        for dtype in [torch.float32, torch.float64]:
            B = random.randrange(1, 10)
            C = random.randrange(1, 10)
            T = random.randrange(1, 20)
            x = torch.randn(B, C, T, dtype=dtype)

            K = 2 ** random.randrange(0, 4)
            N = K * 2
            params = torch.randn(C, N + 1, dtype=dtype)
            x.requires_grad = True
            params.requires_grad = True
            print(f"B,C,T,K = {B},{C},{T},{K}")
            y = learned_nonlin(x, params, dim = 1)


            if torch.cuda.is_available():
                # test that the CUDA forward is the same as the CPU forward.
                device = torch.device('cuda:0')
                y2 = learned_nonlin(x.to(device), params.to(device), dim = 1).to(torch.device('cpu'))
                print("Checking CUDA is same")
                if not torch.allclose(y, y2, atol=1.0e-05):
                    print(f"Error: CPU versus CUDA not the same: {y} vs. {y2}, diff = {y2-y}, max-diff = {(y2-y).abs().max()}")
                    assert(0)

            y_deriv = torch.randn_like(y)
            y.backward(gradient=y_deriv)

            delta = 1.0e-04
            delta_x = torch.randn_like(x) * delta
            pred_change = (x.grad * delta_x).sum()
            y2 = learned_nonlin(x + delta_x, params, dim = 1)
            observed_change = (y_deriv * (y2 - y)).sum()
            print(f"for input: pred_change = {pred_change}, observed_change={observed_change}")
            if not torch.allclose(pred_change, observed_change, rtol=2.0e-02, atol=1.0e-05):
                print(f"For changed input, output differs too much: params={params}, input={x}, mod_input={x+delta_x}, y={y}, y2={y2}, diff={y2-y}")
                assert 0

            delta_params = torch.randn_like(params) * delta
            pred_change = (params.grad * delta_params).sum()
            observed_change = (y_deriv * (learned_nonlin(x, params + delta_params, dim = 1) - y)).sum()
            print(f"for params: pred_change = {pred_change}, observed_change={observed_change}")
            assert torch.allclose(pred_change, observed_change, rtol=1.0e-02, atol=1.0e-05)



def test_learned_nonlin_zeros():
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
            pos_mul = torch.ones(C, kH, kW, device=device, dtype=dtype)
            input.requires_grad = True
            pos_add.requires_grad = True
            pos_mul.requires_grad = True

            output_ref = torch.zeros(N, C, H, W, device=device, dtype=dtype)
            output = learned_nonlin(input, pos_add, pos_mul)
            assert torch.allclose(output, output_ref)

            output.sum().backward()
            print("input_grad=", input.grad)
            print("pos_add_grad=", pos_add.grad)
            print("pos_mul_grad=", pos_mul.grad)


def test_learned_nonlin_compare():
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
        input_cuda = input.to(device).detach()

        kH = 5
        kW = 5
        pos_add = torch.randn(C, kH, kW, dtype=dtype)
        pos_mul = torch.randn(C, kH, kW, dtype=dtype)

        pos_add_cuda = pos_add.to(device).detach()
        pos_mul_cuda = pos_mul.to(device).detach()

        for x in [ pos_add, pos_mul, pos_add_cuda, pos_mul_cuda, input, input_cuda ]:
            x.requires_grad = True

        output = learned_nonlin(input, pos_add, pos_mul)
        output_cuda = learned_nonlin(input_cuda, pos_add_cuda, pos_mul_cuda)
        print("output = ", output)
        print("output_cuda = ", output_cuda)

        output_grad = torch.randn(*output.shape, dtype=dtype)
        output.backward(gradient=output_grad)
        output_cuda.backward(gradient=output_grad.to(device))

        diff = (output - output_cuda.to(torch.device('cpu'))).abs().sum()
        abs = output.abs().sum()
        print("Diff = ", diff, ", abs = ", abs)
        assert torch.allclose(output, output_cuda.to(torch.device('cpu')),
                              atol=1.0e-05)


        for a,b,name in [ (pos_add, pos_add_cuda, 'pos_add'),
                          (pos_mul, pos_mul_cuda, 'pos_mul'),
                          (input, input_cuda, 'input') ]:
            grad = a.grad
            cuda_grad = b.grad.to(torch.device('cpu'))
            diff_abs = (grad - cuda_grad).abs().sum().item()
            sum_abs = (grad + cuda_grad).abs().sum().item()
            print(f"Comparing grad of {name}: diff={diff_abs}, sum={sum_abs}")
            if diff_abs > 1.0e-05 * sum_abs:
                print(f"Error: too much difference in grad of {name}.")
                print("grad = ", grad)
                print("cuda_grad = ", cuda_grad)



def test_learned_nonlin_rand_compare():
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

            output = learned_nonlin(input, pos_add, pos_mul)
            output_cuda = learned_nonlin(input_cuda, pos_add_cuda, pos_mul_cuda)

            diff = (output - output_cuda.to(torch.device('cpu'))).abs().sum()
            sum_abs = output.abs().sum()
            print("Diff = ", diff, ", abs = ", sum_abs)

            if (diff / sum_abs).item() > 0.001:
                print("output = ", output)
                print("output_cuda = ", output_cuda)
                assert 0, "outputs differ"



def test_learned_nonlin_rand_grad():
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

        for device in [ torch.device('cpu'), torch.device('cuda:0') ]:
            if device == torch.device('cuda:0') and not torch.cuda.is_available():
                print("Warning: torch not available, not testing this part.")
                continue
            for dtype in [torch.float32, torch.float64]:
                print("dtype=", dtype, ", device=", device)
                input = torch.randn(N, 2 * C, H, W, dtype=dtype, device=device)


                kH = random.randint(1, 10)
                kW = random.randint(1, 10)
                if kH % 2 == 0:
                    kH += 1
                if kW % 2 == 0:
                    kW += 1
                pos_add = torch.randn(C, kH, kW, dtype=dtype, device=device)
                pos_mul = torch.randn(C, kH, kW, dtype=dtype, device=device)
                input.requires_grad = True
                pos_add.requires_grad = True
                pos_mul.requires_grad = True

                output = learned_nonlin(input, pos_add, pos_mul)
                output_grad = torch.randn(N, C, H, W, dtype=dtype, device=device)

                output.backward(gradient=output_grad)

                delta = 1.0e-05
                pos_delta = delta * torch.randn(C, kH, kW, dtype=dtype, device=device)
                pred_change = (pos_delta * pos_add.grad).sum().to('cpu').item()
                change = (output_grad * (learned_nonlin(input, pos_add + pos_delta, pos_mul) - output )).sum().to('cpu').item()
                print(f"For pos_add: pred_change={pred_change}, change={change}")
                #assert abs(pred_change - change)  < 1.0e-04

                pred_change = (pos_delta * pos_mul.grad).sum().to('cpu').item()
                change = (output_grad * (learned_nonlin(input, pos_add, pos_mul + pos_delta) - output )).sum().to('cpu').item()
                print(f"For pos_mul: pred_change={pred_change}, change={change}")
                #assert abs(pred_change - change) / abs(change) < 1.0e-04

                input_delta = delta * torch.randn(N, 2*C, H, W, dtype=dtype, device=device)
                pred_change = (input_delta * input.grad).sum().to('cpu').item()
                change = (output_grad * (learned_nonlin(input + input_delta, pos_add, pos_mul) - output )).sum().to('cpu').item()
                print(f"For input: pred_change={pred_change}, change={change}")
                #assert abs(pred_change - change) / abs(change) < 1.0e-04


if __name__ == "__main__":
    test_learned_nonlin_basic()
    test_learned_nonlin_deriv()
    if False:
        test_learned_nonlin_rand_grad()
        test_learned_nonlin_zeros()
        test_learned_nonlin_compare()
        test_learned_nonlin_rand_compare()
