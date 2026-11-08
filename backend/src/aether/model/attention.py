import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.config import ModelConfig
from aether.model.embeddings import RotaryPositionalEmbedding

class KVCache:
    """
    pre-allocated Static/Rolling key-Value Cache for O(T) Autoregressive Decoding
    Tracks memory telemetry and sequence offset per layer.
    """
    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.max_seq_len = max_seq_len
        cache_shape = (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.curr_seq_len = 0

    def update(self, xk: torch.Tensor, xv: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Writes new k/v slices into the pre-allocated tensor buffer and returns the active slice.
        If start_pos + seq_len exceeds max_seq_len, performs a rolling window shift.
        """
        bsz, seq_len, _, _ = xk.shape

        if start_pos + seq_len > self.max_seq_len:
            shift = (start_pos + seq_len) - self.max_seq_len
            self.k_cache[:bsz, :-shift] = self.k_cache[:bsz, shift:].clone()
            self.v_cache[:bsz, :-shift] = self.v_cache[:bsz, shift:].clone()
            start_pos = self.max_seq_len - seq_len

        self.k_cache[:bsz, start_pos : start_pos + seq_len] = xk
        self.v_cache[:bsz, start_pos : start_pos + seq_len] = xv
        self.curr_seq_len = start_pos + seq_len

        keys = self.k_cache[:bsz, : self.curr_seq_len]
        values = self.v_cache[:bsz, : self.curr_seq_len]
        return keys, values

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.curr_seq_len = 0

    def active_memory_bytes(self) -> int:
        element_size = self.k_cache.element_size()
        active_elements = (2 * self.k_cache.shape[0] * self.curr_seq_len * self.k_cache.shape[2] * self.k_cache.shape[3])
        return active_elements * element_size

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with RoPE explicit KV_Cache."""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = config.num_queries_per_kv

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _repest_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Expand KV heads from [B, S, n_kv_heads, D] -> [B, S, n_heads, D]
        to match Query head count without duplicating physical weights.
        """
        if n_rep == 1:
            return x
        bsz, seq_len, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bsz, seq_len, n_kv_heads, n_rep, head_dim)
            .expand(bsz, seq_len, n_kv_heads * n_rep, head_dim)
        )
    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding, start_pos: int, kv_cache: Optional[KVCache] = None, use_cacahe: bool = False) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        xq = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = rope(xq, xk, start_pos=start_pos)

        if use_cacahe and kv_cache is not None:
            keys, values = kv_cache.update(xk, xv, start_pos)
        else:
            keys, values = xk, xv

        keys = self._repeat_kv(keys, self.num_queries_per_kv)
        values = self._repeat_kv(values, self.num_queries_per_kv)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if seq_len > 1:
            total_len = keys.shape[2]
            casual_mask = torch.full(
                (seq_len, total_len), float("-inf"), device=x.device, dtype=scores.dtype
            )
            casual_mask = torch.triu(casual_mask, diagonal=1 + start_pos)
            scores = scores + casual_mask.unsqueeze(0).unsqueeze(0)

        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        probs = self.attn_dropout(probs)

        output = torch.matmul(probs, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output)
