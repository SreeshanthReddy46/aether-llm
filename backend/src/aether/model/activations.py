import math
import torch
import torch.nn as nn

class CustomGELU(nn.Module):
    """Explicit Tanh-approximated GELU activation written from first principles."""

    def __init__(self, approximate: bool = True) -> None:
        super().__init__()
        self.approximate = approximate
        self.sqrt_2_over_pi = math.sqrt(2.0 / math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate:
            return 0.5 * x * (1.0 + torch.tanh(self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))))
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GELUFeedForward(nn.Module):
    """standard 2-layer transformer MLP using customGELU."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.act = CustomGELU(approximate=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Gated Feed-Forward Network:
    FFN_SwiGLU(x) = (SiLU(x W_gate) * (x W_up) W-down
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        multiple_of = 64
        adjusted_hidden = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(dim, adjusted_hidden, bias=False)
        self.w_up = nn.Linear(dim, adjusted_hidden, bias=False)
        self.w_down = nn.Linear(adjusted_hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self._silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(gated))
