"""Optional model loading and batch inference for live draft states."""

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.model.features import DraftState, FeatureSpace


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("PyTorch is required to load a trained draft model") from error


class WinPredictor:
    """Batch predictor for blue-side win probabilities using a fitted feature space."""

    def __init__(self, model: Any, feature_space: FeatureSpace) -> None:
        self._model = model
        self.feature_space = feature_space
        self._device: str = "cpu"

    @classmethod
    def load(cls, path: Path) -> "WinPredictor":
        """Load a local model checkpoint and its serialized feature space.
        
        Uses GPU (CUDA) when available, falling back to CPU otherwise.
        """
        torch = _torch_module()
        from app.model.net import DraftNet

        # Detect GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid model checkpoint: {path}")
        try:
            feature_space = FeatureSpace.model_validate(checkpoint["feature_space"])
            input_size = int(checkpoint["input_size"])
            hidden_size = int(checkpoint["hidden_size"])
            dropout = float(checkpoint["dropout"])
            state_dict = checkpoint["state_dict"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid model checkpoint: {path}") from error
        if input_size != feature_space.input_size:
            raise ValueError("Model input size does not match the saved feature space")

        model = DraftNet(input_size, hidden_size=hidden_size, dropout=dropout)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(device)
        predictor = cls(model, feature_space)
        predictor._device = device
        return predictor

    def predict(self, states: Sequence[DraftState]) -> list[float]:
        """Predict blue-side game win probabilities for a batch of draft states.
        
        Uses GPU (CUDA) when available for faster inference.
        """
        if not states:
            return []
        torch = _torch_module()
        features = [self.feature_space.encode(state) for state in states]
        tensor = torch.tensor(features, dtype=torch.float32, device=self._device)
        with torch.inference_mode():
            predictions = self._model(tensor)
        # Move to CPU for Python float conversion if on GPU
        if self._device != "cpu":
            predictions = predictions.cpu()
        return [float(probability) for probability in predictions.tolist()]
