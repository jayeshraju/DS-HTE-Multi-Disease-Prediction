"""
Deep Tabular Learning Modules.
Implements PyTorch-based Tabular Multi-Layer Perceptron (MLP) and FT-Transformer (Feature Tokenizer Transformer).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ContinuousFeatureTokenizer(nn.Module):
    """
    Linear Feature Tokenizer projecting scalar continuous attributes into d-dimensional token embeddings.
    """
    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        self.weight = nn.Parameter(torch.Tensor(num_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(num_features, d_token))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, num_features]
        # output shape: [batch_size, num_features, d_token]
        return x.unsqueeze(-1) * self.weight + self.bias


class PiecewiseLinearTokenizer(nn.Module):
    """
    Piecewise Linear Encoding (PLE) Tokenizer.
    Splits each feature into discrete bins and performs linear interpolation,
    then projects to token embeddings. Capture non-linear threshold mappings.
    """
    def __init__(self, num_features: int, d_token: int, n_bins: int = 8):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        self.n_bins = n_bins
        
        # Bins boundaries initialized uniformly between -3.0 and 3.0
        init_boundaries = torch.linspace(-3.0, 3.0, n_bins + 1).unsqueeze(0).repeat(num_features, 1)
        self.boundaries = nn.Parameter(init_boundaries)
        
        # Learnable projections for bins to tokens: shape [num_features, n_bins, d_token]
        self.weight = nn.Parameter(torch.Tensor(num_features, n_bins, d_token))
        self.bias = nn.Parameter(torch.Tensor(num_features, d_token))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, num_features]
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(-1)  # [batch_size, num_features, 1]

        # boundaries: [1, num_features, n_bins + 1]
        bounds = self.boundaries.unsqueeze(0)
        left = bounds[:, :, :-1]
        right = bounds[:, :, 1:]

        # Piecewise Linear Interpolation
        ple_raw = (x_expanded - left) / (right - left + 1e-6)
        ple = torch.clamp(ple_raw, 0.0, 1.0)  # [batch_size, num_features, n_bins]

        # Compute output embedding via Einstein summation
        tokens = torch.einsum('bnf,nfd->bnd', ple, self.weight) + self.bias.unsqueeze(0)
        return tokens


class FTTransformer(nn.Module):
    """
    FT-Transformer (Feature Tokenizer Transformer) for structured tabular prediction.
    """
    def __init__(
        self,
        num_continuous: int,
        d_token: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        d_ffn_factor: float = 4/3,
        dropout: float = 0.2,
        use_ple: bool = False,
        n_bins: int = 8,
        use_residual_bypass: bool = False
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.d_token = d_token
        self.use_residual_bypass = use_residual_bypass

        # Feature Tokenizer selection
        if use_ple:
            self.tokenizer = PiecewiseLinearTokenizer(num_continuous, d_token, n_bins=n_bins)
        else:
            self.tokenizer = ContinuousFeatureTokenizer(num_continuous, d_token)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=int(d_token * d_ffn_factor),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)
        )

    def forward_features(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        Extracts [CLS] token feature representation z_[CLS] in R^{batch_size x d_token}
        """
        batch_size = x_num.size(0)
        tokens = self.tokenizer(x_num)  # [batch_size, num_continuous, d_token]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_token]
        x = torch.cat([cls_tokens, tokens], dim=1)  # [batch_size, num_continuous + 1, d_token]
        
        z = self.transformer(x)
        cls_repr = z[:, 0, :]  # Extract [CLS] embedding
        
        if self.use_residual_bypass:
            input_mean = tokens.mean(dim=1)
            cls_repr = cls_repr + input_mean
            
        return cls_repr

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        cls_repr = self.forward_features(x_num)
        logits = self.head(cls_repr)
        return torch.sigmoid(logits)
