"""
Tabular Transformer (FT-Transformer style) for clinical risk prediction.
Follows Section III-D:
- Continuous feature tokenizer
- Learnable [CLS] token
- Multi-head self-attention blocks with strong regularization
- Self-supervised masked feature pretraining on training split
- Supervised fine-tuning up to 10 epochs with per-epoch metric logging
"""

import copy
from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin


class TabularFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        # Each numerical feature has its own weight and bias
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_features)
        # Output: (batch_size, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class TabularTransformerNet(nn.Module):
    def __init__(self, n_features: int, d_token: int = 64, n_heads: int = 4,
                 n_layers: int = 2, ffn_mult: int = 2, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = TabularFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        
        # Heads
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)
        )
        self.reconstruction_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)
        )

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        tokens = self.tokenizer(x)  # (B, M, d)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d)
        all_tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, M+1, d)
        encoded = self.transformer(all_tokens)  # (B, M+1, d)
        return self.norm(encoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.forward_tokens(x)
        cls_out = encoded[:, 0, :]  # (B, d)
        logit = self.classifier_head(cls_out).squeeze(-1)
        return logit

    def reconstruct(self, x_masked: torch.Tensor) -> torch.Tensor:
        encoded = self.forward_tokens(x_masked)
        feature_encoded = encoded[:, 1:, :]  # (B, M, d)
        reconstruction = self.reconstruction_head(feature_encoded).squeeze(-1)
        return reconstruction


class TabularTransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.2,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 epochs: int = 10,
                 pretrain_epochs: int = 0,
                 batch_size: int = 32,
                 patience: int = 4,
                 random_state: int = 42,
                 device: Optional[str] = None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_ = None
        self.training_history_ = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[tuple] = None,
            sample_weight: Optional[np.ndarray] = None):
        self._set_seed()
        n_samples, n_features = X.shape

        self.model_ = TabularTransformerNet(
            n_features=n_features,
            d_token=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout
        ).to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        batch_size = min(self.batch_size, n_samples)
        n_batches = int(np.ceil(n_samples / batch_size))

        val_X_t, val_y_t = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_X_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            val_y_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # 1. Optional self-supervised pretraining (Masked Feature Modeling)
        if self.pretrain_epochs > 0:
            pretrain_optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            mse_loss_fn = nn.MSELoss()
            self.model_.train()
            for _ in range(self.pretrain_epochs):
                perm = torch.randperm(n_samples)
                for b_i in range(n_batches):
                    b_idx = perm[b_i * batch_size : (b_i + 1) * batch_size]
                    batch_x = X_t[b_idx]
                    mask = (torch.rand_like(batch_x) < 0.15).float()
                    x_masked = batch_x * (1 - mask)
                    recon = self.model_.reconstruct(x_masked)
                    loss = mse_loss_fn(recon * mask, batch_x * mask)

                    pretrain_optimizer.zero_grad()
                    loss.backward()
                    pretrain_optimizer.step()

        # 2. Supervised Fine-Tuning
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Pos weight for class imbalance if needed
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        self.training_history_ = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_losses, train_correct, total_train = [], 0, 0
            perm = torch.randperm(n_samples)
            for b_i in range(n_batches):
                b_idx = perm[b_i * batch_size : (b_i + 1) * batch_size]
                batch_x = X_t[b_idx]
                batch_y = y_t[b_idx]

                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).float()
                train_correct += (preds == batch_y).sum().item()
                total_train += len(batch_y)

            mean_train_loss = float(np.mean(train_losses))
            mean_train_acc = float(train_correct / max(1, total_train))

            # Validation evaluation
            if val_X_t is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = self.model_(val_X_t)
                    val_loss = criterion(val_logits, val_y_t).item()
                    val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
                    val_acc = (val_preds == val_y_t).sum().item() / len(val_y_t)
            else:
                val_loss = mean_train_loss
                val_acc = mean_train_acc

            self.training_history_["epoch"].append(epoch)
            self.training_history_["train_loss"].append(mean_train_loss)
            self.training_history_["val_loss"].append(val_loss)
            self.training_history_["train_acc"].append(mean_train_acc)
            self.training_history_["val_acc"].append(val_acc)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_t)
            probs_1 = torch.sigmoid(logits).cpu().numpy()
        probs_1 = np.clip(probs_1, 1e-7, 1 - 1e-7)
        probs_0 = 1.0 - probs_1
        return np.column_stack([probs_0, probs_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
