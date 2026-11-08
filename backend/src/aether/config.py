from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    max_batch_size: int = 4
    ffn_mult: float = 2.6875
    activation_type: Literal["swiglu", "gelu"] = "swiglu"
    norm_epos: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0, "model dim must be divisible by n_heads"
        return self.dim // self.n_heads

    @property
    def num_queries_per_kv(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        return self.n_heads // self.n_kv_heads
