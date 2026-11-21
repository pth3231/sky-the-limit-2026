"""PyTorch network used as the draft search fitness function."""

from torch import Tensor, nn


class DraftNet(nn.Module):
    """Two-hidden-layer MLP that predicts blue-side game win probability."""

    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return one blue-side win probability per input row."""
        return self.layers(features).squeeze(-1)
