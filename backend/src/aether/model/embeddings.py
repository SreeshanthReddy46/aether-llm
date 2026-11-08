from typing import Tuple, Any
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square of Normalization:
    y = (x / sqrt(mean(x^2) + eps)) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryPositionalEmbedding(nn.Module):
    """
    Precomputes complex cisoids exp(i * m * theta) and applies rotary position
    encoding to Query and Key tensors across arbitrary KV_cache start offsets.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        freqs = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2)[:head_dim // 2].float() / head_dim)
        )
        t = torch.arange(max_seq_len * 2, dtype=torch.float32)
        freqs_outer = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xq: [batch, seq_len, n_heads, head_dim]
            xk: [batch, seq_len, n_kv_heads, head_dim]
            start_pos: Token offset inside the KV Cache
        """
        seq_len = xq.shape[1]
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        shape = [1, seq_len, 1, xq_complex.shape[-1]]
        freqs_cis = freqs_cis.view(*shape)
        xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
