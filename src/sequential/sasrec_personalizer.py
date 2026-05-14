"""
SASRec Personalizer - Stage 2: Sequential personalization using transformer-based modeling.
Implements simplified SASRec for session-aware recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from ..utils.config_loader import ConfigLoader


class SASRecPersonalizer:
    """
    SASRec-based sequential personalizer.
    
    Implements attention-based sequence modeling for next-item prediction.
    """
    
    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.hidden_size = config.get('sequential.hidden_size', 128)
        self.num_heads = config.get('sequential.num_heads', 4)
        self.num_blocks = config.get('sequential.num_blocks', 2)
        self.max_seq_length = config.get('sequential.max_seq_length', 50)
        self.top_k = config.get('sequential.top_k', 100)
        self.dropout_rate = config.get('sequential.dropout_rate', 0.1)
        
        self.item_embeddings = None
        self.position_embeddings = None
        self.user_sequences = {}
    
    def _create_item_embeddings(self, interactions: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create item embeddings based on co-occurrence."""
        items = interactions['item_id'].unique()
        n_items = len(items)
        
        # Initialize embeddings
        np.random.seed(42)
        embeddings = np.random.normal(0, 0.1, (n_items, self.hidden_size))
        
        # Simple co-occurrence based initialization
        item2idx = {item: i for i, item in enumerate(items)}
        
        # Count co-occurrences
        cooccur = np.zeros((n_items, n_items))
        for user_id, group in interactions.groupby('user_id'):
            user_items = group['item_id'].tolist()
            for i in range(len(user_items) - 1):
                if user_items[i] in item2idx and user_items[i+1] in item2idx:
                    idx_i = item2idx[user_items[i]]
                    idx_j = item2idx[user_items[i+1]]
                    cooccur[idx_i, idx_j] += 1
        
        # Normalize and use as embedding adjustment
        cooccur_norm = cooccur / (cooccur.sum(axis=1, keepdims=True) + 1e-8)
        embeddings = embeddings + 0.1 * cooccur_norm @ embeddings
        
        return {items[i]: embeddings[i] for i in range(n_items)}, item2idx
    
    def _attention_score(self, seq_items: List[str], target_pos: int) -> np.ndarray:
        """Compute attention scores for sequence positions."""
        if not seq_items or not self.item_embeddings:
            return np.array([])
        
        seq_len = min(len(seq_items), self.max_seq_length)
        scores = np.zeros(seq_len)
        
        # Simplified attention: recent items get higher weight
        for i in range(seq_len):
            position_weight = 1.0 / (seq_len - i + 1)  # Recency bias
            scores[i] = position_weight
        
        # Normalize
        scores = scores / (scores.sum() + 1e-8)
        return scores
    
    def _predict_next_item(self, seq_items: List[str]) -> Dict[str, float]:
        """Predict next item probabilities given sequence."""
        if not seq_items or not self.item_embeddings:
            return {}
        
        # Get attention weights
        attention_weights = self._attention_score(seq_items, len(seq_items))
        
        # Compute weighted sum of item embeddings (user representation)
        user_repr = np.zeros(self.hidden_size)
        valid_items = []
        for i, item in enumerate(seq_items[-self.max_seq_length:]):
            if item in self.item_embeddings and i < len(attention_weights):
                user_repr += attention_weights[i] * self.item_embeddings[item]
                valid_items.append(item)
        
        if not valid_items:
            return {}
        
        # L2 normalize
        user_repr = user_repr / (np.linalg.norm(user_repr) + 1e-8)
        
        # Score all items
        scores = {}
        for item_id, emb in self.item_embeddings.items():
            score = float(np.dot(user_repr, emb))
            # Penalize already seen items
            if item_id in seq_items:
                score *= 0.1
            scores[item_id] = score
        
        return scores
    
    def fit(self, sequences_data: Dict[str, Any]) -> 'SASRecPersonalizer':
        """Train SASRec model on sequences."""
        self.logger.info("Training SASRec...")
        
        sequences = sequences_data.get('sequences', {})
        
        # Build item embeddings from all sequences
        all_interactions = []
        for user_id, seq_data in sequences.items():
            items = seq_data.get('items', [])
            for item in items:
                all_interactions.append({'user_id': user_id, 'item_id': item})
        
        if all_interactions:
            df = pd.DataFrame(all_interactions)
            self.item_embeddings, self.item2idx = self._create_item_embeddings(df)
        
        # Store sequences
        self.user_sequences = sequences
        
        self.logger.info(f"Trained on {len(sequences)} sequences, {len(self.item_embeddings)} items")
        return self
    
    def personalize(self, user_id: str, context: Optional[List[str]] = None) -> List[tuple]:
        """Get personalized recommendations for a user."""
        if user_id not in self.user_sequences and not context:
            return []
        
        # Get user's sequence
        if context:
            seq_items = context
        elif user_id in self.user_sequences:
            seq_items = self.user_sequences[user_id].get('items', [])
        else:
            seq_items = []
        
        # Predict
        scores = self._predict_next_item(seq_items)
        
        # Sort and return top-K
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        return sorted_items
    
    def run(self, sequences_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete personalization pipeline."""
        # Train
        self.fit(sequences_data)
        
        # Personalize for all users
        sequences = sequences_data.get('sequences', {})
        all_results = []
        
        for user_id in sequences.keys():
            personalized = self.personalize(user_id)
            for rank, (item_id, score) in enumerate(personalized, 1):
                all_results.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'personalization_score': score,
                    'rank': rank
                })
        
        results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
        
        results = {
            'n_personalized': len(results_df),
            'n_users': len(sequences),
            'top_k': self.top_k,
            'personalized_candidates': results_df
        }
        
        self.logger.info(f"Generated {len(results_df)} personalized candidates")
        return results
