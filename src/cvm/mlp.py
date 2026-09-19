"""
MLP Classifier module for CVM Stage Classification.
Ablation Variant (iii): (-Symbolic, Replaced Rules with MLP)

Redirects the 13 (x, y) coordinates (26 numbers per image) predicted by Model 1
into a lightweight 2-layer MLP classifier to predict CS1-CS6 without vision retraining.
"""

import os
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical CVM Stages (CS1 to CS6)
CVM_STAGE_NAMES: List[str] = ["CS1", "CS2", "CS3", "CS4", "CS5", "CS6"]

# Standard landmark ordering matching Model 1 (13 landmarks)
LANDMARK_NAMES: List[str] = [
    "C2_PI",  # 0: C2 inferior_posterior
    "C2_IC",  # 1: C2 inferior_concavity
    "C2_AI",  # 2: C2 inferior_anterior
    "C3_PS",  # 3: C3 superior_posterior
    "C3_AS",  # 4: C3 superior_anterior
    "C3_PI",  # 5: C3 inferior_posterior
    "C3_IC",  # 6: C3 inferior_concavity
    "C3_AI",  # 7: C3 inferior_anterior
    "C4_PS",  # 8: C4 superior_posterior
    "C4_AS",  # 9: C4 superior_anterior
    "C4_PI",  # 10: C4 inferior_posterior
    "C4_IC",  # 11: C4 inferior_concavity
    "C4_AI",  # 12: C4 inferior_anterior
]


class CVMStageMLP(nn.Module):
    """
    Lightweight 2-layer Neural Network for CVM Stage Classification.
    
    Architecture:
        Linear(26, hidden_dim) -> ReLU -> [optional Dropout] -> Linear(hidden_dim, 6)
        
    Input: 26 numbers (13 x, y coordinates per image)
    Output: 6 class logits (CS1 - CS6)
    """

    def __init__(
        self,
        input_dim: int = 26,
        hidden_dim: int = 64,
        num_classes: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (B, 26) or (26,)
        Returns:
            Logits of shape (B, 6) or (6,)
        """
        is_1d = x.dim() == 1
        if is_1d:
            x = x.unsqueeze(0)
        
        logits = self.net(x)
        
        if is_1d:
            logits = logits.squeeze(0)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices (0 to 5)."""
        logits = self.forward(x)
        if logits.dim() == 1:
            return torch.argmax(logits, dim=0)
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities for classes CS1-CS6."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def coords_to_features(
    data: Union[Any, Dict[str, Any], np.ndarray, torch.Tensor, List]
) -> torch.Tensor:
    """
    Converts diverse landmark data representations into a flat
    26-dimensional float Tensor (B, 26) suitable for CVMStageMLP.
    
    Supported input formats:
      - CVMInput instance
      - Nested dict with 'C2', 'C3', 'C4' (each with landmark point sub-keys)
      - Flat dict with landmark keys (e.g., 'C2_PI', 'C2_IC', ...)
      - Numpy array / list of shape (13, 2) or (26,) or (B, 13, 2) or (B, 26)
      - PyTorch tensor of corresponding shapes
    """
    # 1. If it is already a torch tensor
    if isinstance(data, torch.Tensor):
        t = data.float()
        if t.dim() == 1 and t.numel() == 26:
            return t.unsqueeze(0)
        elif t.dim() == 2:
            if t.shape == (13, 2):
                return t.reshape(1, 26)
            elif t.shape[-1] == 26:
                return t
        elif t.dim() == 3 and t.shape[1:] == (13, 2):
            return t.reshape(t.shape[0], 26)
        raise ValueError(f"Unsupported Tensor shape for landmark coords: {t.shape}")

    # 2. If it is a NumPy array
    if isinstance(data, np.ndarray):
        arr = data.astype(np.float32)
        if arr.ndim == 1 and arr.size == 26:
            return torch.from_numpy(arr).unsqueeze(0)
        elif arr.ndim == 2:
            if arr.shape == (13, 2):
                return torch.from_numpy(arr.reshape(1, 26))
            elif arr.shape[1] == 26:
                return torch.from_numpy(arr)
        elif arr.ndim == 3 and arr.shape[1:] == (13, 2):
            return torch.from_numpy(arr.reshape(arr.shape[0], 26))
        raise ValueError(f"Unsupported NumPy array shape for landmark coords: {arr.shape}")

    # 3. If it is a CVMInput instance
    if hasattr(data, "c2") and hasattr(data, "c3") and hasattr(data, "c4"):
        c2 = data.c2
        c3 = data.c3
        c4 = data.c4
        pts = [
            (c2.inferior_posterior.x, c2.inferior_posterior.y),
            (c2.inferior_concavity.x, c2.inferior_concavity.y),
            (c2.inferior_anterior.x, c2.inferior_anterior.y),
            (c3.superior_posterior.x, c3.superior_posterior.y),
            (c3.superior_anterior.x, c3.superior_anterior.y),
            (c3.inferior_posterior.x, c3.inferior_posterior.y),
            (c3.inferior_concavity.x, c3.inferior_concavity.y),
            (c3.inferior_anterior.x, c3.inferior_anterior.y),
            (c4.superior_posterior.x, c4.superior_posterior.y),
            (c4.superior_anterior.x, c4.superior_anterior.y),
            (c4.inferior_posterior.x, c4.inferior_posterior.y),
            (c4.inferior_concavity.x, c4.inferior_concavity.y),
            (c4.inferior_anterior.x, c4.inferior_anterior.y),
        ]
        flat = [coord for pt in pts for coord in pt]
        return torch.tensor([flat], dtype=torch.float32)

    # 4. If it is a nested dict with 'C2', 'C3', 'C4'
    if isinstance(data, dict) and "C2" in data and "C3" in data and "C4" in data:
        c2 = data["C2"]
        c3 = data["C3"]
        c4 = data["C4"]
        def _get_xy(pt):
            if isinstance(pt, dict):
                return float(pt["x"]), float(pt["y"])
            elif isinstance(pt, (tuple, list)):
                return float(pt[0]), float(pt[1])
            elif hasattr(pt, "x") and hasattr(pt, "y"):
                return float(pt.x), float(pt.y)
            raise ValueError(f"Cannot parse landmark point: {pt}")

        pts = [
            _get_xy(c2["inferior-posterior"]),
            _get_xy(c2["inferior-concavity"]),
            _get_xy(c2["inferior-anterior"]),
            _get_xy(c3["superior-posterior"]),
            _get_xy(c3["superior-anterior"]),
            _get_xy(c3["inferior-posterior"]),
            _get_xy(c3["inferior-concavity"]),
            _get_xy(c3["inferior-anterior"]),
            _get_xy(c4["superior-posterior"]),
            _get_xy(c4["superior-anterior"]),
            _get_xy(c4["inferior-posterior"]),
            _get_xy(c4["inferior-concavity"]),
            _get_xy(c4["inferior-anterior"]),
        ]
        flat = [coord for pt in pts for coord in pt]
        return torch.tensor([flat], dtype=torch.float32)

    # 5. Flat list or tuple
    if isinstance(data, (list, tuple)):
        flat = [float(x) for x in data]
        if len(flat) == 26:
            return torch.tensor([flat], dtype=torch.float32)
        elif len(flat) == 13 and all(isinstance(p, (tuple, list)) and len(p) == 2 for p in data):
            flat_coords = [float(c) for p in data for c in p]
            return torch.tensor([flat_coords], dtype=torch.float32)

    raise ValueError(f"Cannot parse 26 landmark features from input: {type(data)}")


# Global cached default model instance
_CACHED_MLP_MODEL: Optional[CVMStageMLP] = None


def get_default_mlp_model(weights_path: Optional[str] = None) -> CVMStageMLP:
    """
    Returns an initialized CVMStageMLP instance, optionally loading pretrained weights.
    If no weights path is provided, looks in standard locations or initializes cleanly.
    """
    global _CACHED_MLP_MODEL
    if _CACHED_MLP_MODEL is not None and weights_path is None:
        return _CACHED_MLP_MODEL

    model = CVMStageMLP(input_dim=26, hidden_dim=64, num_classes=6)

    # Candidate paths for weights
    search_paths = []
    if weights_path:
        search_paths.append(weights_path)
    search_paths.extend([
        os.path.join(os.path.dirname(__file__), "../../artifacts/mlp_cvm_stage.pt"),
        os.path.join(os.path.dirname(__file__), "../artifacts/mlp_cvm_stage.pt"),
        "artifacts/mlp_cvm_stage.pt",
    ])

    loaded = False
    for path in search_paths:
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location="cpu")
                model.load_state_dict(state_dict)
                loaded = True
                break
            except Exception:
                pass

    model.eval()
    if weights_path is None and loaded:
        _CACHED_MLP_MODEL = model
    return model


def classify_cvm_stage_mlp(
    input_data: Any,
    model: Optional[CVMStageMLP] = None,
    weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classifies the Cervical Vertebral Maturation (CVM) stage (CS1-CS6)
    using the 2-layer MLP classifier instead of the geometric rules.
    
    Ablation Variant (iii): (-Symbolic, Replaced Rules with MLP)
    
    Args:
        input_data: CVMInput, nested dictionary, or array/tensor of 13 landmark coordinates.
        model: Optional CVMStageMLP model instance. If None, default/saved model is used.
        weights_path: Optional path to saved weights.
        
    Returns:
        A dictionary containing:
            - 'stage': The predicted stage string ('CS1' to 'CS6').
            - 'method': 'mlp' (denoting Ablation Variant iii).
            - 'confidence': Prediction confidence probability for the predicted class.
            - 'probabilities': Dict mapping each stage 'CS1'..'CS6' to its predicted probability.
            - 'logits': List of raw class logits.
            - 'details': Metadata regarding the MLP architecture and feature input.
    """
    if model is None:
        model = get_default_mlp_model(weights_path=weights_path)
    model.eval()

    # Extract 26 features
    features = coords_to_features(input_data)  # shape (1, 26)

    with torch.no_grad():
        logits = model(features)[0]  # shape (6,)
        probs = F.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs).item())

    stage = CVM_STAGE_NAMES[pred_idx]
    confidence = float(probs[pred_idx].item())
    probabilities = {
        CVM_STAGE_NAMES[i]: float(probs[i].item()) for i in range(len(CVM_STAGE_NAMES))
    }

    details = {
        "model_architecture": f"CVMStageMLP(Linear(26, {model.hidden_dim}) -> ReLU -> Linear({model.hidden_dim}, 6))",
        "input_features_count": 26,
        "predicted_class_index": pred_idx,
        "ablation": "Model 4: Ablation Variant (iii) (-Symbolic, Replaced Rules with MLP)",
        "coordinates_26": features[0].tolist(),
    }

    return {
        "stage": stage,
        "method": "mlp",
        "confidence": confidence,
        "probabilities": probabilities,
        "logits": logits.tolist(),
        "details": details,
    }
