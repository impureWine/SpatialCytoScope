"""PCA-only supervised reference models without spatial inputs."""

from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


class BaselineSpatialModel:
    """Fit independent domain and cell-mixture predictors using only X_pca."""

    def __init__(self) -> None:
        """Initialize the specified deterministic forest and Ridge regressors."""
        self.classifier = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1)
        self.regressor = MultiOutputRegressor(Ridge(alpha=1.0), n_jobs=1)
        self.is_fitted: bool = False

    @staticmethod
    def _features(adata: ad.AnnData) -> NDArray[np.float32]:
        """Validate the exclusively PCA-based input.

        Args:
            adata: Object containing 30-dimensional PCA features.

        Returns:
            Finite float32 feature matrix.

        Raises:
            ValueError: If PCA features are missing, nonfinite or malformed.
        """
        x = np.asarray(adata.obsm.get("X_pca", []), dtype=np.float32)
        if x.shape != (adata.n_obs, 30) or not np.isfinite(x).all() or adata.n_obs == 0:
            raise ValueError("Require nonempty, finite X_pca with shape (N, 30)")
        return x

    def fit(self, adata: ad.AnnData) -> "BaselineSpatialModel":
        """Fit both supervised estimators on a training slice.

        Args:
            adata: Training PCA features, domain_true and six-column cell truth.

        Returns:
            Fitted instance.

        Raises:
            ValueError: If training annotations or proportions are invalid.
        """
        x = self._features(adata)
        if "domain_true" not in adata.obs or adata.obs["domain_true"].isna().any():
            raise ValueError("Training requires complete domain_true labels")
        y = np.asarray(adata.obsm.get("cell_proportions_true", []), dtype=np.float32)
        if (y.shape != (adata.n_obs, 6) or not np.isfinite(y).all()
                or np.any(y < 0) or not np.allclose(y.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Cell truth must be finite nonnegative (N, 6) rows summing to one")
        self.classifier.fit(x, adata.obs["domain_true"].to_numpy())
        self.regressor.fit(x, y)
        self.is_fitted = True
        return self

    def predict(self, adata: ad.AnnData) -> ad.AnnData:
        """Write domain predictions and simplex-constrained cell mixtures.

        Args:
            adata: Slice with PCA features; annotations are not read.

        Returns:
            Input with the two baseline prediction slots populated.

        Raises:
            RuntimeError: If fit or load has not completed.
        """
        if not self.is_fitted:
            raise RuntimeError("Fit or load the baseline before prediction")
        x = self._features(adata)
        adata.obs["domain_pred_baseline"] = pd.Categorical(self.classifier.predict(x))
        proportions = np.maximum(self.regressor.predict(x), 0).astype(np.float32)
        totals = proportions.sum(axis=1, keepdims=True)
        proportions[totals[:, 0] <= np.finfo(np.float32).eps] = 1 / 6
        proportions /= proportions.sum(axis=1, keepdims=True)
        adata.obsm["cell_proportions_pred_baseline"] = proportions
        return adata

    def save(self, path: Path | str) -> None:
        """Persist fitted estimators to a trusted local joblib artifact.

        Args:
            path: Output artifact filename.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted baseline")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: Path | str) -> "BaselineSpatialModel":
        """Load a locally trusted fitted baseline artifact.

        Args:
            path: File created by save; joblib artifacts must be trusted.

        Returns:
            Fitted baseline model.

        Raises:
            ValueError: If the artifact is not a fitted baseline.
        """
        model = joblib.load(path)
        if not isinstance(model, cls) or not model.is_fitted:
            raise ValueError("Artifact does not contain a fitted BaselineSpatialModel")
        return model
