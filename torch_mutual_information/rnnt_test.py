
import random
import torch
from torch_mutual_information import mutual_information_recursion, joint_mutual_information_recursion, get_rnnt_logprobs, rnnt_loss_simple, rnnt_loss_aux


def test_rnnt_logprobs_basic():
    print("Running test_rnnt_logprobs_basic()")

    B = 1
    S = 3
    T = 4
    C = 3

    # lm: [B][S+1][C]
    lm = torch.tensor([[[ 0, 0, 1 ], [0, 1, 1], [1, 0, 1], [2, 2, 0]]], dtype=torch.float)
    # am: [B][T][C]
    am = torch.tensor([[[ 0, 1, 2], [0, 0, 0 ], [0, 2, 4 ], [0, 3, 3]]], dtype=torch.float)

#    lm[:] = 0.0
#    am[:] = 0.0

    termination_symbol = 2
    symbols = torch.tensor([[ 0, 1, 0 ] ], dtype=torch.long)

    px, py = get_rnnt_logprobs(lm, am, symbols, termination_symbol)

    assert px.shape == (B, S, T+1)
    assert py.shape == (B, S+1, T)
    assert symbols.shape == (B, S)
    print("px = ", px)
    print("py = ", py)
    m = mutual_information_recursion(px, py)
    print("m = ", m)


    # should be invariant to adding a constant for any frame.
    lm += torch.randn(B, S+1, 1)
    am += torch.randn(B, T, 1)

    m2 = rnnt_loss_simple(lm, am, symbols, termination_symbol, None)
    print("m2 = ", m2)

    device = torch.device('cuda')
    m3 = rnnt_loss_simple(lm.to(device), am.to(device), symbols.to(device), termination_symbol, None)
    print("m3 = ", m3)

    device = torch.device('cuda')
    m4 = rnnt_loss_aux(lm.to(device), am.to(device), symbols.to(device), termination_symbol,
                       lm_only_scale=0.0, am_only_scale=0.0, boundary=None)
    print("m4 = ", m4)

    assert torch.allclose(m, m2)

    assert torch.allclose(m, m3.to('cpu'))

    assert torch.allclose(m, m4.to('cpu'))

def test_rnnt_logprobs_aux():

    print("Running test_rnnt_logprobs_aux()")

    B = 1
    S = 3
    T = 4
    C = 3

    # lm: [B][S+1][C]
    lm = torch.tensor([[[ 0, 0, 1 ], [0, 1, 1], [1, 0, 1], [2, 2, 0]]], dtype=torch.float)
    # am: [B][T][C]
    am = torch.tensor([[[ 0, 1, 2], [0, 0, 0 ], [0, 2, 4 ], [0, 3, 3]]], dtype=torch.float)

    termination_symbol = 2
    symbols = torch.tensor([[ 0, 1, 0 ] ], dtype=torch.long)


    device = torch.device('cuda')
    m1 = rnnt_loss_aux(lm.to(device), am.to(device), symbols.to(device), termination_symbol,
                       lm_only_scale=0.0, am_only_scale=0.333, boundary=None)
    print("m1 = ", m1)


    # should be invariant to adding a constant for any frame.
    lm += torch.randn(B, S+1, 1)
    am += torch.randn(B, T, 1)

    m2 = rnnt_loss_aux(lm.to(device), am.to(device), symbols.to(device), termination_symbol,
                       lm_only_scale=0.0, am_only_scale=0.333, boundary=None)
    print("m2 = ", m2)


    assert torch.allclose(m1, m2)



if __name__ == "__main__":
    #torch.set_printoptions(edgeitems=30)
    test_rnnt_logprobs_aux()
    test_rnnt_logprobs_basic()
