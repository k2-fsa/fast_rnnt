
import random
import torch
from torch_mutual_information import mutual_information_recursion, joint_mutual_information_recursion, get_rnnt_logprobs, rnnt_loss_simple, rnnt_loss


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
    m, grad = mutual_information_recursion(px, py)
    print("m = ", m)

    t_am = am.unsqueeze(2)
    t_lm = lm.unsqueeze(1)
    t_prob = t_am + t_lm
    m1, grad = rnnt_loss(t_prob, symbols, termination_symbol, None)
    print ("m1 = ", m1)
    assert torch.allclose(m, m1)

    # should be invariant to adding a constant for any frame.
    lm += torch.randn(B, S+1, 1)
    am += torch.randn(B, T, 1)

    m2, grad = rnnt_loss_simple(lm, am, symbols, termination_symbol, None)
    print("m2 = ", m2)
    assert torch.allclose(m, m2)

    t_am = am.unsqueeze(2)
    t_lm = lm.unsqueeze(1)
    t_prob = t_am + t_lm
    m3, grad = rnnt_loss(t_prob, symbols, termination_symbol, None)
    print ("m3 = ", m3)
    assert torch.allclose(m, m3)


if __name__ == "__main__":
    #torch.set_printoptions(edgeitems=30)
    test_rnnt_logprobs_basic()
