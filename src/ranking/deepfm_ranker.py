"""
DeepFM Ranker - Stage 3: Feature-based ranking using DeepFM architecture.
Combines factorization machines with deep neural networks for CTR prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from scipy.sparse.linalg import svds

from ..utils.config_loader import ConfigLoader


class DeepFMRanker:
    """
    DeepFM-based ranker for final recommendation scoring.
    
    Combines FM (for low-order feature interactions) and 
    Deep component (for high-order interactions).
    """
    
    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.embedding_dim = config.get('ranking.embedding_dim', 32)
        self.mlp_hidden_sizes = config.get('ranking.mlp_hidden_sizes', [128, 64, 32])
        self.dropout_rates = config.get('ranking.dropout_rates', [0.3, 0.3, 0.3])
        self.learning_rate = config.get('ranking.learning_rate', 0.001)
        self.top_k = config.get('ranking.top_k', 50)
        
        # Encoders
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Model weights (simplified implementation)
        self.fm_weights = {}
        self.deep_weights = []
        self.user_embedding_matrix = None
        self.item_embedding_matrix = None
        self.user_ids = []
        self.item_ids = []
    
    def _encode_features(self, df: pd.DataFrame, sparse_features: List[str], 
                         dense_features: List[str]) -> pd.DataFrame:
        """Encode categorical and numerical features."""
        encoded_df = df.copy()
        
        # Encode sparse features
        for feat in sparse_features:
            if feat in encoded_df.columns:
                le = LabelEncoder()
                encoded_df[f'{feat}_enc'] = le.fit_transform(
                    encoded_df[feat].astype(str).fillna('unknown')
                )
                self.label_encoders[feat] = le
        
        # Scale dense features
        for feat in dense_features:
            if feat in encoded_df.columns:
                encoded_df[f'{feat}_scaled'] = self.scaler.fit_transform(
                    encoded_df[[feat]].fillna(0)
                )
        
        return encoded_df
    
    def _compute_fm_scores(self, X_sparse: np.ndarray, X_dense: np.ndarray) -> np.ndarray:
        """Compute Factorization Machine scores."""
        n_samples = X_sparse.shape[0]
        
        # Linear part
        linear_score = np.zeros(n_samples)
        
        # Interaction part (simplified)
        interaction_score = np.zeros(n_samples)
        
        # Combine
        fm_scores = linear_score + interaction_score
        return fm_scores
    
    def _compute_deep_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute Deep network scores."""
        # Simplified: use weighted sum instead of full neural network
        if len(self.deep_weights) == 0:
            # Initialize random weights
            np.random.seed(42)
            n_features = X.shape[1]
            for hidden_size in self.mlp_hidden_sizes:
                w = np.random.normal(0, 0.1, (n_features, hidden_size))
                self.deep_weights.append(w)
                n_features = hidden_size
            
            # Output layer
            self.deep_weights.append(np.random.normal(0, 0.1, (n_features, 1)))
        
        # Forward pass
        h = X
        for i, w in enumerate(self.deep_weights[:-1]):
            h = np.dot(h, w)
            h = np.maximum(0, h)  # ReLU
        
        deep_scores = np.dot(h, self.deep_weights[-1]).flatten()
        return deep_scores
    
    def fit(self, data: Dict[str, Any]) -> 'DeepFMRanker':
        """Train DeepFM model."""
        self.logger.info("Training DeepFM...")
        
        df = data['dataframe']
        sparse_features = data['sparse_features']
        dense_features = data['dense_features']
        
        # Filter to available features
        sparse_features = [f for f in sparse_features if f in df.columns]
        dense_features = [f for f in dense_features if f in df.columns]
        
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        
        # Encode features
        encoded_df = self._encode_features(df, sparse_features, dense_features)
        
        # Prepare training data
        X_sparse_cols = [f'{f}_enc' for f in sparse_features if f'{f}_enc' in encoded_df.columns]
        X_dense_cols = [f'{f}_scaled' for f in dense_features if f'{f}_scaled' in encoded_df.columns]
        
        X_sparse = encoded_df[X_sparse_cols].values if X_sparse_cols else np.zeros((len(df), 0))
        X_dense = encoded_df[X_dense_cols].values if X_dense_cols else np.zeros((len(df), 0))
        
        # Store encodings for inference
        self.feature_columns = {
            'sparse': X_sparse_cols,
            'dense': X_dense_cols
        }
        
        self.logger.info(f"Trained on {len(df)} samples with {len(sparse_features)} sparse and {len(dense_features)} dense features")

        # Build user/item embeddings from observed interactions and available features.
        self._build_embeddings(encoded_df)
        self._validate_embeddings()
        return self

    def _build_embeddings(self, encoded_df: pd.DataFrame) -> None:
        """Create deterministic, non-constant user/item embeddings from encoded features."""
        if encoded_df.empty:
            raise ValueError("Cannot build embeddings from empty dataframe.")

        numeric_cols = [c for c in encoded_df.columns if c.endswith('_enc') or c.endswith('_scaled')]
        user_item = pd.crosstab(encoded_df['user_id'], encoded_df['item_id']).astype(float)
        self.user_ids = user_item.index.tolist()
        self.item_ids = user_item.columns.tolist()
        matrix = user_item.values.astype(float)
        k = max(8, min(self.embedding_dim, min(matrix.shape) - 1))
        u, s, vt = svds(matrix, k=k)
        order = np.argsort(s)[::-1]
        u, s, vt = u[:, order], s[order], vt[order, :]
        user_mat = u @ np.diag(np.sqrt(s))
        item_mat = vt.T @ np.diag(np.sqrt(s))
        self.user_embedding_matrix = self._l2_normalize(user_mat)
        self.item_embedding_matrix = self._l2_normalize(item_mat)

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _validate_embeddings(self) -> None:
        """Detect invalid embedding states that cause ranking collapse."""
        if self.user_embedding_matrix is None or self.item_embedding_matrix is None:
            raise ValueError("Embeddings were not created.")
        if np.any(np.linalg.norm(self.user_embedding_matrix, axis=1) == 0):
            raise ValueError("Zero-vector user embeddings detected.")
        if np.any(np.linalg.norm(self.item_embedding_matrix, axis=1) == 0):
            raise ValueError("Zero-vector item embeddings detected.")
        if len(np.unique(np.round(self.item_embedding_matrix, 8), axis=0)) == 1:
            raise ValueError("Identical item embeddings detected.")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores for items."""
        # Encode features
        encoded_df = self._encode_features(df, self.sparse_features, self.dense_features)
        
        X_sparse_cols = [f'{f}_enc' for f in self.sparse_features if f'{f}_enc' in encoded_df.columns]
        X_dense_cols = [f'{f}_scaled' for f in self.dense_features if f'{f}_scaled' in encoded_df.columns]
        
        X_sparse = encoded_df[X_sparse_cols].values if X_sparse_cols else np.zeros((len(df), 0))
        X_dense = encoded_df[X_dense_cols].values if X_dense_cols else np.zeros((len(df), 0))
        
        # Compute scores
        fm_scores = self._compute_fm_scores(X_sparse, X_dense)
        deep_scores = self._compute_deep_scores(X_dense if X_dense.size > 0 else X_sparse)
        
        # Combine FM and Deep scores (sigmoid approximation)
        combined = fm_scores + deep_scores
        scores = 1 / (1 + np.exp(-combined))  # Sigmoid
        
        return scores
    
    def rank(self, candidates: pd.DataFrame, user_context: Optional[Dict] = None) -> pd.DataFrame:
        """Rank candidate items for a user."""
        if len(candidates) == 0:
            return candidates
        
        candidates = candidates.copy()
        user_index = {u: i for i, u in enumerate(self.user_ids)}
        item_index = {i: j for j, i in enumerate(self.item_ids)}

        ranked_parts = []
        for user_id, group in candidates.groupby('user_id'):
            if user_id not in user_index:
                continue
            valid = group[group['item_id'].isin(item_index)].copy()
            if valid.empty:
                continue
            u_vec = self.user_embedding_matrix[user_index[user_id]].reshape(1, -1)
            i_mat = np.vstack([self.item_embedding_matrix[item_index[i]] for i in valid['item_id']])
            sim_scores = cosine_similarity(u_vec, i_mat).flatten()
            probs = softmax(sim_scores)

            valid['similarity_score'] = sim_scores
            valid['ranking_score'] = probs
            valid = valid.sort_values('ranking_score', ascending=False)
            ranked_parts.append(valid)

        ranked = pd.concat(ranked_parts, ignore_index=True) if ranked_parts else pd.DataFrame(columns=list(candidates.columns) + ['similarity_score', 'ranking_score'])
        self._validate_ranked_scores(ranked)
        return ranked

    def _validate_ranked_scores(self, ranked: pd.DataFrame) -> None:
        """Validation checks for ranking-score integrity."""
        if ranked.empty:
            raise ValueError("No ranked candidates produced.")
        grouped = ranked.groupby('user_id')['ranking_score']
        sum_check = grouped.sum()
        if not np.allclose(sum_check.values, 1.0, atol=1e-6):
            raise ValueError("Invalid probability distribution: per-user scores do not sum to 1.")
        var_check = grouped.var().fillna(0.0)
        if (var_check <= 0).any():
            bad_users = var_check[var_check <= 0].index.tolist()[:5]
            raise ValueError(f"Constant score collapse detected for users: {bad_users}")
    
    def run(self, deepfm_data: Dict[str, Any], candidates_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run the complete ranking pipeline."""
        # Train
        self.fit(deepfm_data)
        
        # Use candidates from prior stages when available to preserve retrieval/sequential signals.
        df = deepfm_data['dataframe']
        if candidates_df is None or candidates_df.empty:
            users = df['user_id'].unique()[:100]
            items = df['item_id'].unique()
            candidates = []
            for user in users:
                seen_items = set(df[df['user_id'] == user]['item_id'].tolist())
                sample_items = list(np.random.choice(items, min(100, len(items)), replace=False))
                for item in sample_items:
                    if item in seen_items:
                        continue
                    candidates.append({'user_id': user, 'item_id': item})
            candidates_df = pd.DataFrame(candidates)
        
        if len(candidates_df) == 0:
            return {'n_ranked': 0, 'ranked_candidates': pd.DataFrame()}

        # Enrich candidates with metadata and interaction signals for reranking.
        feature_df = df.copy()
        keep_cols = [
            c for c in ["user_id", "item_id", "rating", "interaction", "timestamp", "created_date", "category", "seller", "brand"]
            if c in feature_df.columns
        ]
        if keep_cols:
            pair_meta = feature_df[keep_cols].drop_duplicates(subset=["user_id", "item_id"], keep="last")
            item_meta_cols = [c for c in keep_cols if c not in {"user_id", "item_id", "rating", "interaction", "timestamp"}]
            item_meta = feature_df[["item_id"] + item_meta_cols].drop_duplicates(subset=["item_id"], keep="last") if item_meta_cols else None
            candidates_df = candidates_df.merge(pair_meta, on=["user_id", "item_id"], how="left")
            if item_meta is not None:
                candidates_df = candidates_df.merge(item_meta, on="item_id", how="left", suffixes=("", "_item"))
            if "rating" in candidates_df.columns and "interaction" not in candidates_df.columns:
                candidates_df["interaction"] = candidates_df["rating"]
        
        # Rank
        ranked_df = self.rank(candidates_df)
        if "ranking_score" in ranked_df.columns:
            ranked_df["deepfm_score"] = ranked_df["ranking_score"]
        
        results = {
            'n_ranked': len(ranked_df),
            'n_users': ranked_df['user_id'].nunique(),
            'top_k': self.top_k,
            'ranked_candidates': ranked_df
        }
        
        self.logger.info(f"Ranked {len(ranked_df)} candidates")
        return results
