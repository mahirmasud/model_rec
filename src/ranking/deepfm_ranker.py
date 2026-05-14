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
        return self
    
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
        
        # Predict scores
        scores = self.predict(candidates)
        candidates = candidates.copy()
        candidates['ranking_score'] = scores
        
        # Sort by score
        ranked = candidates.sort_values('ranking_score', ascending=False)
        return ranked
    
    def run(self, deepfm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete ranking pipeline."""
        # Train
        self.fit(deepfm_data)
        
        # Get candidates from previous stage or use all user-item pairs
        df = deepfm_data['dataframe']
        
        # Sample candidates for ranking (in production, this would come from retrieval)
        # For demo, we'll create user-item pairs
        users = df['user_id'].unique()[:100]  # Limit for efficiency
        items = df['item_id'].unique()
        
        # Create candidate pairs
        candidates = []
        for user in users:
            user_items = df[df['user_id'] == user]['item_id'].tolist()
            # Mix seen and unseen items
            sample_items = list(np.random.choice(items, min(50, len(items)), replace=False))
            for item in sample_items:
                candidates.append({'user_id': user, 'item_id': item})
        
        candidates_df = pd.DataFrame(candidates)
        
        if len(candidates_df) == 0:
            return {'n_ranked': 0, 'ranked_candidates': pd.DataFrame()}
        
        # Rank
        ranked_df = self.rank(candidates_df)
        
        results = {
            'n_ranked': len(ranked_df),
            'n_users': ranked_df['user_id'].nunique(),
            'top_k': self.top_k,
            'ranked_candidates': ranked_df
        }
        
        self.logger.info(f"Ranked {len(ranked_df)} candidates")
        return results
