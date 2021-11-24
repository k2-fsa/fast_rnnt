import os

import torch
from torch import Tensor
from typing import Tuple, Optional
from . mutual_information import mutual_information_recursion, joint_mutual_information_recursion



def get_rnnt_logprobs(lm: Tensor,
                      am: Tensor,
                      symbols: Tensor,
                      termination_symbol: int) -> Tuple[Tensor, Tensor]:
    """
    Reduces RNN-T problem (the simple case, where joiner network is just addition),
    to a compact, standard form that can then be given
    (with boundaries) to mutual_information_recursion().  This function is called from
    rnnt_loss_simple(), but may be useful for other purposes.

    Args:
         lm:  Language model part of un-normalized logprobs of symbols, to be added to
              acoustic model part before normalizing.  Of shape:
                 [B][S+1][C]
              where B is the batch size, S is the maximum sequence length of
              the symbol sequence, possibly including the EOS symbol; and
              C is size of the symbol vocabulary, including the termination/next-frame
              symbol.
              Conceptually, lm[b][s] is a vector of length [C] representing the
              "language model" part of the un-normalized logprobs of symbols,
              given all symbols *earlier than* s in the sequence.  The reason
              we still need this for position S is that we may still be emitting
              the termination/next-frame symbol at this point.
         am:  Acoustic-model part of un-normalized logprobs of symbols, to be added
              to language-model part before normalizing.  Of shape:
                 [B][T][C]
              where B is the batch size, T is the maximum sequence length of
              the acoustic sequences (in frames); and C is size of the symbol
              vocabulary, including the termination/next-frame symbol.  It reflects
              the "acoustic" part of the probability of any given symbol appearing
              next on this frame.
          symbols: A LongTensor of shape [B][S], containing the symbols at each position
              of the sequence, possibly including EOS
          termination_symbol: The identity of the termination symbol, must be
               in {0..C-1}
    Returns: (px, py) (the names are quite arbitrary).
              px: logprobs, of shape [B][S][T+1]
              py: logprobs, of shape [B][S+1][T]
          in the recursion:
             p[b,0,0] = 0.0
             p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                p[b,s,t-1] + py[b,s,t-1])
          .. where p[b][s][t] is the "joint score" of the pair of subsequences of
          length s and t respectively.  px[b][s][t] represents the probability of
          extending the subsequences of length (s,t) by one in the s direction,
          given the particular symbol, and py[b][s][t] represents the probability
          of extending the subsequences of length (s,t) by one in the t direction,
          i.e. of emitting the termination/next-frame symbol.

          px[:,:,T] equals -infinity, meaning on the "one-past-the-last" frame
          we cannot emit any symbols.  This is simply a way of incorporating
          the probability of the termination symbol on the last frame.
    """
    assert lm.ndim== 3 and am.ndim == 3 and lm.shape[0] == am.shape[0] and lm.shape[2] == am.shape[2]
    (B, T, C) = am.shape
    S = lm.shape[1] - 1
    assert symbols.shape == (B, S)

    # subtracting am_max and lm_max is to ensure the probs are in a good range to do exp()
    # without causing underflow or overflow.
    am_max, _ = torch.max(am, dim=2, keepdim=True)  # am_max: [B][T][1]
    lm_max, _ = torch.max(lm, dim=2, keepdim=True)  # lm_max: [B][S+1][1]
    am_probs = (am - am_max).exp()
    lm_probs = (lm - lm_max).exp()
    # normalizers: [B][S+1][T]
    normalizers = (torch.matmul(lm_probs, am_probs.transpose(1, 2)) + 1.0e-20).log()

    # add lm_max and am_max to normalizers, to make it as if we had not
    # subtracted am_max and lm_max above.
    normalizers = normalizers + lm_max + am_max.transpose(1, 2)  # [B][S+1][T]

    # px is the probs of the actual symbols..
    px_am = torch.gather(am.unsqueeze(1).expand(B, S, T, C), dim=3,
                         index=symbols.reshape(B, S, 1, 1).expand(B, S, T, 1)).squeeze(-1) # [B][S][T]
    px_am = torch.cat((px_am,
                       torch.full((B, S, 1), float('-inf'),
                                  device=px_am.device, dtype=px_am.dtype)),
                      dim=2)  # now: [B][S][T+1], index [:,:,T] has -inf..

    px_lm = torch.gather(lm[:,:S], dim=2, index=symbols.unsqueeze(-1)) # [B][S][1]

    px = px_am + px_lm  # [B][S][T+1], last slice indexed [:,:,T] is -inf
    px[:,:,:T] -= normalizers[:,:S,:] # px: [B][S][T+1]

    # py is the probs of termination symbols, of shape [B][S+1][T]
    py_am = am[:,:,termination_symbol].unsqueeze(1) # [B][1][T]
    py_lm = lm[:,:,termination_symbol].unsqueeze(2) # [B][S+1][1]
    py = py_am + py_lm - normalizers

    return (px, py)


def rnnt_loss_simple(lm: Tensor,
                     am: Tensor,
                     symbols: Tensor,
                     termination_symbol: int,
                     boundary: Tensor = None) -> Tensor:
    """
    A simple case of the RNN-T loss, where the 'joiner' network is just addition.
    Returns total loss value.

    Args:
     lm: language-model part of unnormalized log-probs of symbols, with shape
        (B, S+1, C), i.e. batch, symbol_seq_len+1, num_classes
     am: acoustic-model part of unnormalized log-probs of symbols, with shape
       (B, T, C), i.e. batch, frame, num_classes
     symbols: the symbol sequences, a LongTensor of shape [B][S], and elements in {0..C-1}.
     termination_symbol: the termination symbol, with 0 <= termination_symbol < C
     boundary: a LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
   Returns:
      a Tensor of shape (B,), containing the total RNN-T loss values for each element
      of the batch (like log-probs of sequences).
    """
    px, py = get_rnnt_logprobs(lm, am, symbols, termination_symbol)
    return mutual_information_recursion(px, py, boundary)
