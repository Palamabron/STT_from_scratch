from __future__ import annotations

import torch
import torch.nn.functional as F

from SpeechToText.models.tdt.loss import compute_tdt_losses

__all__ = ["compute_tdt_losses", "_masked_uniform_kl"]


def _masked_uniform_kl(
    *,
    token_logits: torch.Tensor,
    out_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    exclude_blank: bool = True,
) -> torch.Tensor:
    """Uniform KL over valid joint positions; computed per utterance to limit peak memory."""
    device = token_logits.device
    batch_size = int(token_logits.size(0))
    vocab_size = int(token_logits.size(-1))
    total = torch.zeros((), device=device)
    count = 0

    vocab_mask: torch.Tensor | None = None
    effective_vocab = vocab_size
    if exclude_blank and 0 <= blank_id < vocab_size and vocab_size > 1:
        vocab_mask = torch.ones(vocab_size, device=device, dtype=torch.bool)
        vocab_mask[blank_id] = False
        effective_vocab = int(vocab_mask.sum().item())
        if effective_vocab <= 0:
            return torch.zeros((), device=device)

    inv_v = 1.0 / float(effective_vocab)

    for b in range(batch_size):
        t_max = min(int(out_lengths[b].item()), int(token_logits.size(1)))
        u_max = min(int(target_lengths[b].item()) + 1, int(token_logits.size(2)))
        if t_max <= 0 or u_max <= 0:
            continue
        chunk_t, chunk_u = 16, 32
        for t0 in range(0, t_max, chunk_t):
            t1 = min(t0 + chunk_t, t_max)
            for u0 in range(0, u_max, chunk_u):
                u1 = min(u0 + chunk_u, u_max)
                log_p = F.log_softmax(token_logits[b, t0:t1, u0:u1], dim=-1)
                if vocab_mask is not None:
                    log_p = log_p[..., vocab_mask]
                flat = log_p.reshape(-1, effective_vocab)
                uniform = flat.new_full((flat.size(0), effective_vocab), inv_v)
                total = total + F.kl_div(flat, uniform, reduction="sum", log_target=False)
                count += flat.size(0)

    if count == 0:
        return torch.zeros((), device=device)
    return total / float(count)
