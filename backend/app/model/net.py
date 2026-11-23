"""PyTorch network used as the draft search fitness function."""

import torch
from torch import Tensor, nn


class DraftNet(nn.Module):
    """MLP over a self-attention-pooled draft sequence and continuous aggregate features."""

    def __init__(
        self,
        num_champions: int,
        continuous_size: int,
        sequence_length: int,
        *,
        embedding_dim: int = 16,
        hidden_size: int = 32,
        dropout: float = 0.5,
        attention_heads: int = 2,
    ) -> None:
        super().__init__()
        self.champion_embedding = nn.Embedding(num_champions + 1, embedding_dim)
        self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        input_size = embedding_dim + continuous_size
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

    def forward(self, draft_sequence: Tensor, continuous: Tensor) -> Tensor:
        """Return one blue-side win probability per input row."""
        positions = torch.arange(draft_sequence.shape[1], device=draft_sequence.device)
        embedded = self.champion_embedding(draft_sequence) + self.position_embedding(positions)
        attended, _ = self.attention(embedded, embedded, embedded)
        sequence_repr = self.attention_norm(embedded + attended)
        pooled = sequence_repr.mean(dim=1)
        combined = torch.cat([pooled, continuous], dim=1)
        return self.layers(combined).squeeze(-1)
