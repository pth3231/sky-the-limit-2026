"""PyTorch network used as the draft search fitness function."""

import torch
from torch import Tensor, nn


class DraftNet(nn.Module):
    """MLP over mean-pooled champion embeddings and continuous aggregate features."""

    def __init__(
        self,
        num_champions: int,
        continuous_size: int,
        *,
        embedding_dim: int = 16,
        hidden_size: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.champion_embedding = nn.Embedding(num_champions + 1, embedding_dim)
        input_size = 4 * embedding_dim + continuous_size
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

    def forward(
        self,
        blue_picks: Tensor,
        blue_bans: Tensor,
        red_picks: Tensor,
        red_bans: Tensor,
        continuous: Tensor,
    ) -> Tensor:
        """Return one blue-side win probability per input row."""
        pooled = [
            self.champion_embedding(group).mean(dim=1)
            for group in (blue_picks, blue_bans, red_picks, red_bans)
        ]
        combined = torch.cat([*pooled, continuous], dim=1)
        return self.layers(combined).squeeze(-1)
