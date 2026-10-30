"""
Hybrid Deep Attentive Gradient Boosting (HDA-GB) Classifier.
Implements the hybrid architecture coupling FT-Transformer latent [CLS] embeddings with downstream XGBoost / CatBoost.
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from src.models.deep_learning import FTTransformer

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False


def enable_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()


class HDAGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Deep Attentive Gradient Boosting Classifier.
    1. Extracts latent [CLS] embeddings from pre-trained FT-Transformer.
    2. Augments raw features with [CLS] latent embeddings.
    3. Trains downstream CatBoost/XGBoost classifier on the augmented feature matrix.
    """
    def __init__(
        self,
        d_token: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        transformer_epochs: int = 20,
        gbdt_n_estimators: int = 100,
        gbdt_type: str = "catboost",
        class_weights: Optional[List[float]] = None,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        learning_rate: float = 0.03,
        use_ple: bool = False,
        n_bins: int = 8,
        use_residual_bypass: bool = False,
        val_ratio: float = 0.15,
        patience: int = 5,
        use_mc_dropout: bool = False,
        mc_samples: int = 20,
        bypass_transformer: bool = False,
        random_state: int = 42
    ):
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.transformer_epochs = transformer_epochs
        self.gbdt_n_estimators = gbdt_n_estimators
        self.gbdt_type = gbdt_type
        self.class_weights = class_weights
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.learning_rate = learning_rate
        self.use_ple = use_ple
        self.n_bins = n_bins
        self.use_residual_bypass = use_residual_bypass
        self.val_ratio = val_ratio
        self.patience = patience
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples
        self.bypass_transformer = bypass_transformer
        self.random_state = random_state

        self.transformer_model: Optional[FTTransformer] = None
        self.gbdt_model: Optional[Any] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        num_continuous = X.shape[1]
        
        # Check if we bypass the transformer
        bypass = getattr(self, "bypass_transformer", False)
        if bypass:
            self.transformer_model = None
            X_augmented = X
        else:
            # 1. Instantiate FT-Transformer Encoder
            torch.manual_seed(self.random_state)
            self.transformer_model = FTTransformer(
                num_continuous=num_continuous,
                d_token=self.d_token,
                n_blocks=self.n_blocks,
                n_heads=self.n_heads,
                use_ple=self.use_ple,
                n_bins=self.n_bins,
                use_residual_bypass=self.use_residual_bypass
            )

            # Split training and validation sets for early stopping
            if self.val_ratio > 0 and len(y) > 20:
                from sklearn.model_selection import train_test_split
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=self.val_ratio, stratify=y, random_state=self.random_state
                )
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
            else:
                X_tr, y_tr = X, y
                X_val, y_val = None, None

            # Pre-training step for FT-Transformer using BCE loss with early stopping
            self.transformer_model.train()
            optimizer = torch.optim.AdamW(self.transformer_model.parameters(), lr=1e-4, weight_decay=1e-2)
            criterion = torch.nn.BCELoss()

            X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
            y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

            best_val_loss = float('inf')
            best_weights = None
            patience_counter = 0

            for epoch in range(self.transformer_epochs):
                self.transformer_model.train()
                optimizer.zero_grad()
                preds = self.transformer_model(X_tr_tensor)
                loss = criterion(preds, y_tr_tensor)
                loss.backward()
                optimizer.step()

                # Early stopping validation check
                if X_val is not None:
                    self.transformer_model.eval()
                    with torch.no_grad():
                        val_preds = self.transformer_model(X_val_tensor)
                        val_loss = criterion(val_preds, y_val_tensor).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = {k: v.cpu().clone() for k, v in self.transformer_model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            break

            # Load best weights if early stopping was active
            if best_weights is not None:
                self.transformer_model.load_state_dict(best_weights)

            # 2. Extract Latent [CLS] Tokens (use eval mode without MC dropout during fitting)
            self.transformer_model.eval()
            X_all_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                cls_embeddings = self.transformer_model.forward_features(X_all_tensor).numpy()

            # 3. Construct Augmented Feature Space
            X_augmented = np.hstack([X, cls_embeddings])

        # 4. Fit Downstream Gradient Booster
        scale_pos_weight = float(self.class_weights[1] / self.class_weights[0]) if self.class_weights is not None else None

        if self.gbdt_type == "stacking":
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            estimators = []
            if CB_AVAILABLE:
                estimators.append(('catboost', cb.CatBoostClassifier(
                    iterations=self.gbdt_n_estimators,
                    depth=self.depth,
                    l2_leaf_reg=self.l2_leaf_reg,
                    learning_rate=self.learning_rate,
                    random_seed=self.random_state,
                    verbose=False
                )))
            if XGB_AVAILABLE:
                estimators.append(('xgboost', xgb.XGBClassifier(
                    n_estimators=self.gbdt_n_estimators,
                    max_depth=self.depth,
                    reg_lambda=self.l2_leaf_reg,
                    learning_rate=self.learning_rate,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state
                )))
            try:
                from lightgbm import LGBMClassifier
                estimators.append(('lightgbm', LGBMClassifier(
                    n_estimators=self.gbdt_n_estimators,
                    max_depth=self.depth,
                    learning_rate=self.learning_rate,
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state,
                    verbosity=-1
                )))
            except ImportError:
                pass
                
            self.gbdt_model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(C=1.0, class_weight='balanced' if self.class_weights is not None else None, random_state=self.random_state),
                cv=3,
                n_jobs=1
            )
        elif self.gbdt_type == "catboost" and CB_AVAILABLE:
            self.gbdt_model = cb.CatBoostClassifier(
                iterations=self.gbdt_n_estimators,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                learning_rate=self.learning_rate,
                random_seed=self.random_state,
                class_weights=self.class_weights,
                verbose=False
            )
        elif XGB_AVAILABLE:
            self.gbdt_model = xgb.XGBClassifier(
                n_estimators=self.gbdt_n_estimators,
                max_depth=self.depth,
                reg_lambda=self.l2_leaf_reg,
                learning_rate=self.learning_rate,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.gbdt_model = GradientBoostingClassifier(
                n_estimators=self.gbdt_n_estimators,
                random_state=self.random_state
            )

        self.gbdt_model.fit(X_augmented, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.gbdt_model is None:
            raise ValueError("HDAGBClassifier is not fitted.")

        bypass = getattr(self, "bypass_transformer", False)
        if bypass or self.transformer_model is None:
            return self.gbdt_model.predict_proba(X)

        if self.use_mc_dropout:
            # Keep dropout active in the FT-Transformer during inference
            self.transformer_model.eval()
            self.transformer_model.apply(enable_dropout)
            
            probs_list = []
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            for _ in range(self.mc_samples):
                with torch.no_grad():
                    cls_embeddings = self.transformer_model.forward_features(X_tensor).numpy()
                X_augmented = np.hstack([X, cls_embeddings])
                probs_list.append(self.gbdt_model.predict_proba(X_augmented))
                
            return np.mean(probs_list, axis=0)
        else:
            self.transformer_model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                cls_embeddings = self.transformer_model.forward_features(X_tensor).numpy()

            X_augmented = np.hstack([X, cls_embeddings])
            return self.gbdt_model.predict_proba(X_augmented)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Computes prediction confidence score.
        If MC Dropout is active:
          C_pred = (1 - 2 * epistemic_uncertainty) * 2 * |p_mean - 0.5|
        Otherwise:
          C_pred = 2 * |p - 0.5|
        """
        bypass = getattr(self, "bypass_transformer", False)
        if bypass or self.transformer_model is None:
            proba = self.predict_proba(X)[:, 1]
            return 2.0 * np.abs(proba - 0.5)

        if self.use_mc_dropout:
            self.transformer_model.eval()
            self.transformer_model.apply(enable_dropout)
            
            probs_list = []
            X_tensor = torch.tensor(X, dtype=torch.float32)
            for _ in range(self.mc_samples):
                with torch.no_grad():
                    cls_embeddings = self.transformer_model.forward_features(X_tensor).numpy()
                X_augmented = np.hstack([X, cls_embeddings])
                probs_list.append(self.gbdt_model.predict_proba(X_augmented)[:, 1])
                
            probs_arr = np.array(probs_list)  # [mc_samples, batch_size]
            mean_probs = np.mean(probs_arr, axis=0)
            epistemic = np.std(probs_arr, axis=0)
            
            # Scale to range [0, 1]
            raw_conf = 2.0 * np.abs(mean_probs - 0.5)
            # Penalize confidence by 2x the standard deviation (which maxes out at 0.5)
            calibrated_conf = np.maximum(0.0, (1.0 - 2.0 * epistemic) * raw_conf)
            return calibrated_conf
        else:
            proba = self.predict_proba(X)[:, 1]
            return 2.0 * np.abs(proba - 0.5)


class CalibratedThresholdClassifierWrapper:
    def __init__(self, base_model, calibrator, threshold):
        self.base_model = base_model
        self.calibrator = calibrator
        self.threshold = threshold

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        raw_prob = self.base_model.predict_proba(X)[:, 1]
        cal_prob = self.calibrator.predict_proba(raw_prob)[:, 1]
        return np.vstack([1.0 - cal_prob, cal_prob]).T

    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def predict_confidence(self, X):
        return self.base_model.predict_confidence(X)


class BetaCalibrator:
    """
    Beta Calibration for binary classification probability outputs.
    Fits a logistic regression on [ln(p), -ln(1 - p)].
    """
    def __init__(self):
        self.lr = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)

    def fit(self, probs: np.ndarray, y: np.ndarray):
        eps = 1e-7
        p = np.clip(probs, eps, 1.0 - eps)
        X_trans = np.column_stack([np.log(p), -np.log(1.0 - p)])
        self.lr.fit(X_trans, y)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-7
        p = np.clip(probs, eps, 1.0 - eps)
        X_trans = np.column_stack([np.log(p), -np.log(1.0 - p)])
        return self.lr.predict_proba(X_trans)
