"""Toy transformer: just large enough to produce real loss/perplexity
numbers for the Learning Ledger -- not a scale target (see
DATALOADER_DESIGN.md §8 scope assumptions: "Model is a small toy
transformer, only large enough to produce real loss/perplexity numbers
for the learning ledger").

Deliberately small and manually written (no nn.MultiheadAttention, no
attention library) so the two things this whole session has been building
toward -- segment-aware attention and position-id resets -- are visible
and auditable directly in the forward pass, not hidden inside a library
call:

- `position_id` (already reset to 0 at each segment boundary by the
  Packer) indexes the position embedding table directly, rather than
  absolute position within the packed window -- a new document partway
  through a window gets its own local position sequence, not a huge
  meaningless offset inherited from whatever came before it.
- Attention bias is built from `segment_id` via
  `tds.packer.attention_bias_from_segments` -- the same verification
  utility the Packer already exposes, now put to real use as the actual
  attention mask a forward pass applies, not just a test helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch_assembler import Microbatch
from .packer import attention_bias_from_segments

NEG_BIAS = -1e9  # additive attention bias for disallowed positions


@dataclass
class ToyTransformerConfig:
    vocab_size: int
    max_sequence_length: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 128
    dropout: float = 0.0


class _SelfAttention(nn.Module):
    def __init__(self, config: ToyTransformerConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(f"d_model={config.d_model} must be divisible by n_heads={config.n_heads}")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, n_heads, L, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores + attn_bias  # (B, 1, L, L) broadcasts across heads
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.out(out)


class _Block(nn.Module):
    def __init__(self, config: ToyTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = _SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), attn_bias))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, config: ToyTransformerConfig, seed: int = 0):
        # Deterministic init: seeds torch's *global* RNG before building any
        # parameters. A real multi-model/concurrent setup would need
        # generator-scoped init instead -- fine here, where exactly one toy
        # model exists per process.
        torch.manual_seed(seed)
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.d_model)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.n_layers))
        self.ln_final = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self, token_ids: torch.Tensor, position_id: torch.Tensor, attn_bias: torch.Tensor
    ) -> torch.Tensor:
        x = self.token_embedding(token_ids) + self.position_embedding(position_id)
        for block in self.blocks:
            x = block(x, attn_bias)
        x = self.ln_final(x)
        return self.output_projection(x)


def attention_bias_for_microbatch(microbatch: Microbatch) -> torch.Tensor:
    """(B, 1, L, L) additive float bias -- 0 where a token may attend, a
    large negative number where it may not -- derived directly from
    segment_id via the same attention_bias_from_segments the Packer's own
    tests use to verify correctness. This is the O(L^2) materialization
    DATALOADER_DESIGN.md §5.6 says real attention should avoid at scale;
    at this toy model's scale it's the simplest correct thing to do."""
    bool_mask = np.stack([attention_bias_from_segments(seg) for seg in microbatch.segment_id])
    bias = np.where(bool_mask, 0.0, NEG_BIAS).astype(np.float32)
    return torch.from_numpy(bias).unsqueeze(1)  # (B, 1, L, L)


def forward_and_masked_loss(model: ToyTransformer, microbatch: Microbatch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Graph-preserving forward pass + masked next-token loss. Returns
    (masked_loss, mask), both (B, sequence_length - 1) tensors, gradients
    intact -- shared by compute_batch_loss (read-only callers) and
    tds.training_step's backward pass (which needs the graph to survive)."""
    token_ids = torch.from_numpy(microbatch.token_ids)
    position_id = torch.from_numpy(microbatch.position_id)
    attn_bias = attention_bias_for_microbatch(microbatch)

    logits = model(token_ids, position_id, attn_bias)
    shift_logits = logits[:, :-1, :]
    shift_targets = token_ids[:, 1:]

    raw_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_targets.reshape(-1), reduction="none"
    ).view(shift_targets.shape)

    mask = torch.from_numpy(microbatch.loss_mask[:, :-1])
    return raw_loss * mask, mask


def compute_batch_loss(model: ToyTransformer, microbatch: Microbatch) -> Tuple[np.ndarray, float]:
    """Read-only forward pass. Returns (per_token_loss: np.ndarray of
    shape (B, sequence_length - 1), avg_loss: float).

    per_token_loss[:, i] is the loss for predicting token_ids[:, i+1] from
    position i, already multiplied by loss_mask[:, i] -- position
    sequence_length - 1 is dropped entirely (it has no next-token target
    by construction; loss_mask there is always 0 anyway), so summing and
    dividing by the mask's count gives the correctly-masked average.
    """
    masked_loss, mask = forward_and_masked_loss(model, microbatch)
    denom = mask.sum().clamp(min=1)
    avg_loss = (masked_loss.sum() / denom).item()
    return masked_loss.detach().numpy(), avg_loss
