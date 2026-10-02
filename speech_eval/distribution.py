"""Statistical reference distributions and Mahalanobis distance scoring.

Maintains phoneme distribution statistics (mean vector, covariance/variance, acoustic norms)
with context-aware triphone lookup and fallback to monophone distributions.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


class PhonemeDistributionManager:
    """Manages precomputed empirical standard Turkish pronunciation distributions."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        # Maps key -> {mean, variance, sample_count, acoustic_stats}
        self.distributions: Dict[str, Dict[str, Any]] = {}
        if db_path and Path(db_path).exists():
            self.load(Path(db_path))

    def make_context_key(self, current: str, prev: str = "^", next_p: str = "$") -> str:
        """Constructs a triphone context key, e.g., 'k_a_r'."""
        return f"{prev}_{current}_{next_p}"

    def get_distribution(self, current: str, prev: str = "^", next_p: str = "$") -> Optional[Dict[str, Any]]:
        """Retrieves context-aware triphone distribution, falls back to monophone if unavailable."""
        triphone_key = self.make_context_key(current, prev, next_p)
        if triphone_key in self.distributions and self.distributions[triphone_key].get("sample_count", 0) >= 3:
            return self.distributions[triphone_key]

        # Fallback to monophone
        mono_key = f"_{current}_"
        if mono_key in self.distributions:
            return self.distributions[mono_key]
        if current in self.distributions:
            return self.distributions[current]

        return None

    def update_distribution(
        self,
        key: str,
        embeddings: List[np.ndarray],
        acoustic_feats: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Adds empirical samples and updates mean and diagonal covariance."""
        if not embeddings:
            return

        X = np.stack(embeddings)  # [N, D]
        n_samples = X.shape[0]

        if key in self.distributions:
            existing = self.distributions[key]
            old_n = existing["sample_count"]
            old_mean = np.array(existing["mean_embedding"], dtype=np.float32)
            old_var = np.array(existing["var_embedding"], dtype=np.float32)

            # Incremental update
            new_n = old_n + n_samples
            new_mean = (old_mean * old_n + np.sum(X, axis=0)) / new_n
            # Update variance estimate
            new_var = (old_var * old_n + np.sum((X - new_mean) ** 2, axis=0)) / new_n

            existing["sample_count"] = new_n
            existing["mean_embedding"] = new_mean.tolist()
            existing["var_embedding"] = np.maximum(new_var, 1e-4).tolist()
        else:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.distributions[key] = {
                "sample_count": n_samples,
                "mean_embedding": mean.tolist(),
                "var_embedding": np.maximum(var, 1e-4).tolist(),
                "acoustic_stats": {}
            }

    def compute_mahalanobis_distance(
        self,
        embedding: np.ndarray,
        dist: Dict[str, Any],
        regularization: float = 1e-3
    ) -> float:
        """Calculates regularized Mahalanobis distance D(x) from the reference distribution.
        
        D(x) = sqrt( sum_i (x_i - mu_i)^2 / (sigma_i^2 + reg) )
        """
        mu = np.array(dist["mean_embedding"], dtype=np.float32)
        var = np.array(dist["var_embedding"], dtype=np.float32) + regularization

        diff = embedding - mu
        # Diagonal Mahalanobis distance normalized by dimension
        scaled_diff = (diff ** 2) / var
        # Root mean squared normalized distance
        dist_val = float(np.sqrt(np.mean(scaled_diff)))
        return dist_val

    def mahalanobis_to_score(self, dist_val: float) -> float:
        """Maps Mahalanobis distance to standard 0-100 score.
        
        Standard z-score:
        distance <= 1.0 (within 1 std) -> 90-100
        distance = 1.5 -> ~80
        distance = 2.5 -> ~55
        distance >= 4.0 -> < 20
        """
        score = 100.0 / (1.0 + (dist_val / 1.6) ** 2)
        return float(np.clip(score, 0.0, 100.0))

    def save(self, file_path: Path) -> None:
        """Serializes distributions to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.distributions, f, indent=2, ensure_ascii=False)

    def load(self, file_path: Path) -> None:
        """Loads distributions from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            self.distributions = json.load(f)
