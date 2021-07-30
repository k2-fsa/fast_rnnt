# Caution: this will fail occasionally due to cutoffs not being quite large enough.
# As long as it passes most of the time, it's OK.

import random
import torch
from torch_mutual_information import mutual_information_recursion


def test_mutual_information_basic():
    print("Running test_mutual_information_basic()")
    for dtype in [torch.float32, torch.float64]:
        for device in [ torch.device('cpu'), torch.device('cuda:0') ]:
            print("dtype = ", dtype, ", device = ", device)
            B = 2
            S = 33
            T = 33
            boundary = torch.tensor([ 0, 0, S, T ], dtype=torch.int64).unsqueeze(0).expand(B, 4).to(device)
            px = torch.zeros(B, S, T + 1, dtype=dtype).to(device)  # log of an odds ratio
            py = torch.zeros(B, S + 1, T, dtype=dtype).to(device)  # log of an odds ratio

            m = mutual_information_recursion(px, py, None)
            #m = mutual_information_recursion(px, py, boundary)

            print("m = ", m, ", size = ", m.shape)
            print("exp(m) = ", m.exp())



def test_mutual_information_deriv():
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
            y = mutual_information(x, params, dim = 1)

            y_deriv = torch.randn_like(y)
            y.backward(gradient=y_deriv)

            if torch.cuda.is_available():
                # test that the CUDA forward is the same as the CPU forward.
                device = torch.device('cuda:0')
                x2, params2 = x.to(device).detach(), params.to(device).detach()
                x2.requires_grad = True
                params2.requires_grad = True
                y2 = mutual_information(x2, params2, dim = 1)

                if N >= 4 and N <= 16:  # Currently backprop requires these conditions
                    y2.backward(gradient=y_deriv.to(device))
                    x2grad, params2grad = x2.grad.to('cpu'), params2.grad.to('cpu')

                y2 = y2.to('cpu')

                print("Checking CUDA is same")
                if not torch.allclose(y, y2, atol=1.0e-05):
                    print(f"Error: CPU output versus CUDA not the same: {y} vs. {y2}, diff = {y2-y}, max-diff = {(y2-y).abs().max()}")
                    assert(0)

                if N >= 4 and N <= 16:  # Currently backprop requires these conditions
                    if not torch.allclose(x.grad, x2grad, atol=1.0e-05):
                        print(f"Error: CPU x.grad versus CUDA not the same: {x.grad} vs. {x2grad}, diff = {x2grad-x.grad}, max-diff = {(x2grad-x.grad).abs().max()}")
                        assert(0)
                    if not torch.allclose(params.grad, params2grad, atol=1.0e-05):
                        print(f"Error: CPU params.grad versus CUDA not the same: {params.grad} vs. {params2grad}, "
                              f"diff = {params2grad-params.grad}, max-diff = {(params2grad-params.grad).abs().max()}")
                        assert(0)


            delta = 1.0e-04
            delta_x = torch.randn_like(x) * delta
            pred_change = (x.grad * delta_x).sum()
            y2 = mutual_information(x + delta_x, params, dim = 1)
            observed_change = (y_deriv * (y2 - y)).sum()
            print(f"for input: pred_change = {pred_change}, observed_change={observed_change}")
            if not torch.allclose(pred_change, observed_change, rtol=5.0e-02, atol=3.0e-05):
                print(f"For changed input, output differs too much: params={params}, input={x}, mod_input={x+delta_x}, y={y}, y2={y2}, diff={y2-y}")
                assert 0

            delta_params = torch.randn_like(params) * delta
            pred_change = (params.grad * delta_params).sum()
            observed_change = (y_deriv * (mutual_information(x, params + delta_params, dim = 1) - y)).sum()
            print(f"for params: pred_change = {pred_change}, observed_change={observed_change}")
            assert torch.allclose(pred_change, observed_change, rtol=1.0e-02, atol=1.0e-05)



def test_mutual_information_zeros():
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
            output = mutual_information(input, pos_add, pos_mul)
            assert torch.allclose(output, output_ref)

            output.sum().backward()
            print("input_grad=", input.grad)
            print("pos_add_grad=", pos_add.grad)
            print("pos_mul_grad=", pos_mul.grad)


def test_mutual_information_compare():
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

        output = mutual_information(input, pos_add, pos_mul)
        output_cuda = mutual_information(input_cuda, pos_add_cuda, pos_mul_cuda)
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



def test_mutual_information_rand_compare():
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

            output = mutual_information(input, pos_add, pos_mul)
            output_cuda = mutual_information(input_cuda, pos_add_cuda, pos_mul_cuda)

            diff = (output - output_cuda.to(torch.device('cpu'))).abs().sum()
            sum_abs = output.abs().sum()
            print("Diff = ", diff, ", abs = ", sum_abs)

            if (diff / sum_abs).item() > 0.001:
                print("output = ", output)
                print("output_cuda = ", output_cuda)
                assert 0, "outputs differ"



def test_mutual_information_rand_grad():
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

                output = mutual_information(input, pos_add, pos_mul)
                output_grad = torch.randn(N, C, H, W, dtype=dtype, device=device)

                output.backward(gradient=output_grad)

                delta = 1.0e-05
                pos_delta = delta * torch.randn(C, kH, kW, dtype=dtype, device=device)
                pred_change = (pos_delta * pos_add.grad).sum().to('cpu').item()
                change = (output_grad * (mutual_information(input, pos_add + pos_delta, pos_mul) - output )).sum().to('cpu').item()
                print(f"For pos_add: pred_change={pred_change}, change={change}")
                #assert abs(pred_change - change)  < 1.0e-04

                pred_change = (pos_delta * pos_mul.grad).sum().to('cpu').item()
                change = (output_grad * (mutual_information(input, pos_add, pos_mul + pos_delta) - output )).sum().to('cpu').item()
                print(f"For pos_mul: pred_change={pred_change}, change={change}")
                #assert abs(pred_change - change) / abs(change) < 1.0e-04

                input_delta = delta * torch.randn(N, 2*C, H, W, dtype=dtype, device=device)
                pred_change = (input_delta * input.grad).sum().to('cpu').item()
                change = (output_grad * (mutual_information(input + input_delta, pos_add, pos_mul) - output )).sum().to('cpu').item()
                print(f"For input: pred_change={pred_change}, change={change}")
                #assert abs(pred_change - change) / abs(change) < 1.0e-04


if __name__ == "__main__":
    test_mutual_information_basic()
    test_mutual_information_deriv()
    if False:
        test_mutual_information_rand_grad()
        test_mutual_information_zeros()
        test_mutual_information_compare()
        test_mutual_information_rand_compare()
