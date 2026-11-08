import time
import torch
from typing import Dict, Generator, List, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F

from aether.config import ModelConfig
from aether.model.activations import GELUFeedForward, SwiGLUFeedForward
from aether.model.attention import GroupedQueryAttention, KVCache
from aether.model.embeddings import RMSNorm, RotaryPositionalEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_epos)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_epos)

        hidden_dim = int(config.dim * config.ffn_mult)
        if config.activation_type == "swiglu":
            self.ffn = SwiGLUFeedForward(config.dim, hidden_dim, config.dropout)
        else:
            self.ffn = GELUFeedForward(config.dim, hidden_dim, config.dropout)

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding, start_pos: int = 0, kv_cache: Optional[KVCache] = None, use_cache: bool = False) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), rope=rope, start_pos=start_pos, kv_cache=kv_cache, use_cache=use_cache)
        out = h + self.ffn(self.ffn_norm(h))
        return out

class AetherTransformerLM(nn.Module):
    """
    Full Casual Language Model supporting:
    - SwiGLU & GELU Switchable Activations
    - RoPE Positional Encoding
    - Multi-Layer KV-Caching
    - Hidden-State Extraction for Bi-Encoder Embeddings & Reward Modeling
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.rope = RotaryPositionalEmbedding(head_dim=config.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_epos)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.lm_head.weight = self.tok_embeddings.weight
        self.kv_caches: List[KVCache] = []
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_kv_cache(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        target_device = device or next(self.parameters()).device
        self.kv_caches = [
            KVCache(
                max_batch_size=batch_size,
                max_seq_len=self.config.max_seq_len,
                n_kv_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                device=target_device
            )
            for _ in range(self.config.n_layers)
        ]

    def reset_kv_cache(self) -> None:
        for cache in self.kv_caches:
            cache.reset()

    def get_kv_cache_stats(self) -> Dict[str, float]:
        if not self.kv_caches:
            return {"cached_tokens": 0, "memory_kb": 0.0}
        total_bytes = sum(c.active_memory_bytes() for c in self.kv_caches)
        return {"cached_token": self.kv_caches[0].current_seq_len, "memory_kb": round(total_bytes / 1024.0, 2),}

