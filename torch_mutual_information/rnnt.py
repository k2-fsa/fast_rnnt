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
    Returns negated total loss value.

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
      a Tensor of shape (B,), containing the NEGATED total RNN-T loss values
      for each element of the batch (like log-probs of sequences).
    """
    px, py = get_rnnt_logprobs(lm, am, symbols, termination_symbol)
    return mutual_information_recursion(px, py, boundary)


def get_rnnt_logprobs_aux(lm: Tensor,
                          am: Tensor,
                          symbols: Tensor,
                          termination_symbol: int,
                          lm_only_scale: float = 0.1,
                          am_only_scale: float = 0.1) -> Tuple[Tensor, Tensor]:
    """
    Reduces RNN-T problem (the simple case, where joiner network is just addition),
    to a compact, standard form that can then be given
    (with boundaries) to mutual_information_recursion().   This version allows you
    to make the loss-function one of the form:
          lm_only_scale * lm_probs +
          am_only_scale * am_probs +
          (1-lm_only_scale-am_only_scale) * combined_probs
    where lm_probs and am_probs are the probabilities given the lm and acoustic model
    independently.

    This function is called from
    rnnt_loss_aux(), but may be useful for other purposes.

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
          am_only: same shape as am, [B][T][C], but differently normalized or differently
             projected,
             so that it can be interpreted as probabilities without taking the LM into
             account.  Does not have to already include the logsoftmax(); we will do that.
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

    # Caution: some parts of this code are a little less clear than they could
    # be due to optimizations.  In particular it may not be totally obvious that
    # all of the logprobs here are properly normalized.  We test that
    # this code is invariant to adding constants in the appropriate ways.

    # subtracting am_max and lm_max is to ensure the probs are in a good range to do exp()
    # without causing underflow or overflow.
    am_max, _ = torch.max(am, dim=2, keepdim=True)  # am_max: [B][T][1]
    lm_max, _ = torch.max(lm, dim=2, keepdim=True)  # lm_max: [B][S+1][1]
    am_probs = (am - am_max).exp()  # [B][T][C]
    lm_probs = (lm - lm_max).exp()  # [B][S+1][C]
    # normalizers: [B][S+1][T]
    normalizers = (torch.matmul(lm_probs, am_probs.transpose(1, 2)) + 1.0e-20).log()

    # normalizer per frame, if we take only the LM probs by themselves
    lmonly_normalizers = lm_probs.sum(dim=2, keepdim=True) # lmonly_normalizers: [B][S+1][1]
    unigram_lm = torch.mean(lm_probs / lmonly_normalizers, dim=(0,1), keepdim=True) + 1.0e-20 # [1][1][C]
    amonly_normalizers = torch.mv(am_probs.reshape(-1, C), unigram_lm.reshape(C)).reshape(B, T, 1).log() + am_max # [B][T][1]
    amonly_normalizers = amonly_normalizers.transpose(1, 2)  # [B][1][T]
    unigram_lm = unigram_lm.log()
    lmonly_normalizers = lmonly_normalizers.log() + lm_max # [B][S+1][1], log-normalizer, used for LM-only part of prob.


    # add lm_max and am_max to normalizers, to make it as if we had not
    # subtracted am_max and lm_max above.
    normalizers = normalizers + lm_max + am_max.transpose(1, 2)  # [B][S+1][T]

    # px is the probs of the actual symbols (not yet normalized)..
    px_am = torch.gather(am.unsqueeze(1).expand(B, S, T, C), dim=3,
                         index=symbols.reshape(B, S, 1, 1).expand(B, S, T, 1)).squeeze(-1) # [B][S][T]
    px_am = torch.cat((px_am,
                       torch.full((B, S, 1), float('-inf'),
                                  device=px_am.device, dtype=px_am.dtype)),
                      dim=2)  # now: [B][S][T+1], index [:,:,T] has -inf..


    px_lm = torch.gather(lm[:,:S], dim=2, index=symbols.unsqueeze(-1)) # [B][S][1]
    px_lm_unigram = torch.gather(unigram_lm.expand(B, S, C), dim=2, index=symbols.unsqueeze(-1)) # [B][S][1]

    px = px_am + px_lm  # [B][S][T+1], last slice indexed [:,:,T] is -inf
    px[:,:,:T] -= normalizers[:,:S,:] # px: [B][S][T+1]

    px_amonly = px_am + px_lm_unigram     # [B][S][T+1]
    px_amonly[:,:,:T] -= amonly_normalizers
    px_lmonly = px_lm - lmonly_normalizers[:,:S,:]


    # py is the probs of termination symbols, of shape [B][S+1][T]
    py_am = am[:,:,termination_symbol].unsqueeze(1) # [B][1][T]
    py_lm = lm[:,:,termination_symbol].unsqueeze(2) # [B][S+1][1]
    py = py_am + py_lm - normalizers

    py_lm_unigram = unigram_lm[0][0][termination_symbol] # scalar, normalized..
    py_amonly = py_am + py_lm_unigram - amonly_normalizers # [B][S+1][T]
    py_lmonly = py_lm - lmonly_normalizers # [B][S+1][T]


    combined_scale = 1.0 - lm_only_scale - am_only_scale

    # We need to avoid exact zeros in the scales because otherwise multiplying -inf
    # by zero generates nan.
    if lm_only_scale == 0.0:
        lm_only_scale = 1.0e-20
    if am_only_scale == 0.0:
        am_only_scale = 1.0e-20

    px_interp = px * combined_scale + px_lmonly * lm_only_scale + px_amonly * am_only_scale
    py_interp = py * combined_scale + py_lmonly * lm_only_scale + py_amonly * am_only_scale

    print("px_interp = ", px_interp)
    print("py_interp = ", py_interp)
    return (px_interp, py_interp)


def rnnt_loss_aux(lm: Tensor,
                  am: Tensor,
                  symbols: Tensor,
                  termination_symbol: int,
                  lm_only_scale: float = 0.1,
                  am_only_scale: float = 0.1,
                  boundary: Tensor = None) -> Tensor:
    """
    A simple case of the RNN-T loss, where the 'joiner' network is just addition.
    Returns negated total loss value.

    Args:
     lm: language-model part of unnormalized log-probs of symbols, with shape
        (B, S+1, C), i.e. batch, symbol_seq_len+1, num_classes.
        These are assumed to be well-normalized, in the sense that we could
        use them as probabilities separately from the am scores
     am: acoustic-model part of unnormalized log-probs of symbols, with shape
       (B, T, C), i.e. batch, frame, num_classes
     symbols: the symbol sequences, a LongTensor of shape [B][S], and elements in {0..C-1}.
     termination_symbol: the termination symbol, with 0 <= termination_symbol < C
     am_only_scale: the scale on the "AM-only" part of the loss, for which we use
       an "averaged" LM (averaged over all histories, so effectively unigram).
     boundary: a LongTensor of shape [B, 4] with elements interpreted as
        [begin_symbol, begin_frame, end_symbol, end_frame] that is treated as [0, 0, S, T]
        if boundary is not supplied.
        Most likely you will want begin_symbol and begin_frame to be zero.
   Returns:
      a Tensor of shape (B,), containing the NEGATED total RNN-T loss values
      for each element of the batch (like log-probs of sequences).
    """
    px, py = get_rnnt_logprobs_aux(lm, am, symbols, termination_symbol,
                                   lm_only_scale, am_only_scale)
    return mutual_information_recursion(px, py, boundary)
