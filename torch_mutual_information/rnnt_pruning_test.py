import k2
import torch

from torch.nn.functional import sigmoid
from torch_mutual_information import get_pruning_ranges, rnnt_loss, pruning, pruning_rnnt_loss, rnnt_loss_simple

def test_rnnt_pruning():
    device = torch.device("cpu")
    B = 4
    T = 300
    S = 50
    C = 10
    am = torch.randn((B, T, C), dtype=torch.float64, device=device) 
    lm = torch.randn((B, S + 1, C), dtype=torch.float64, device=device)
    symbols = torch.randint(0, C, (B, S), device=device)
    terminal_symbol = C - 1
    print ("(B, T, S, C): ", (B, T, S, C))

    # normal rnnt
    t_am = am.unsqueeze(2).float()
    t_lm = lm.unsqueeze(1).float()
    t_prob = t_am + t_lm
    # nonlinear transform
    t_prob = sigmoid(t_prob)
    k2_loss, grads = rnnt_loss(t_prob, symbols, terminal_symbol, None)
    print ("unpruned rnnt loss: ", k2_loss)
    
    # pruning
    k2_simple_loss, (px_grad, py_grad) = rnnt_loss_simple(lm, am, symbols, terminal_symbol, None)
    for r in range(2, 52, 2):
        ranges = get_pruning_ranges(px_grad, py_grad, r)
        # (B, T, r, C)
        am_p, lm_p = pruning(am, lm, ranges)

        t_prob_p = am_p + lm_p

        # nonlinear transform
        t_prob_p = sigmoid(t_prob_p)
        boundary = torch.zeros((B, 4), dtype=torch.int64)
        boundary[:,2] = ranges[:,-1,-1]
        boundary[:,3] = T

        pruning_loss, grads = pruning_rnnt_loss(t_prob_p, symbols, ranges, terminal_symbol, boundary)
        print (f"pruning loss with range {r} : ", pruning_loss)


if __name__== '__main__':
    test_rnnt_pruning()

