"""
LightGCN Retriever - Stage 1: Candidate retrieval using graph-based collaborative filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from scipy import sparse

from ..utils.config_loader import ConfigLoader


class LightGCNRetriever:
    """LightGCN-based candidate retriever for large-scale recommendation."""
    
    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.embedding_dim = config.get('retrieval.embedding_dim', 64)
        self.n_layers = config.get('retrieval.n_layers', 3)
        self.top_k = config.get('retrieval.top_k', 500)
        
        self.user2id = {}
        self.item2id = {}
        self.id2user = {}
        self.id2item = {}
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_col = 'USER_ID'
        self.item_col = 'ITEM_ID'
    
    def _build_mappings(self, interactions: pd.DataFrame) -> None:
        """Build user and item ID mappings."""
        # Detect column names
        self.user_col = 'user_id' if 'user_id' in interactions.columns else \
                       ('USER_ID' if 'USER_ID' in interactions.columns else interactions.columns[0])
        self.item_col = 'item_id' if 'item_id' in interactions.columns else \
                       ('ITEM_ID' if 'ITEM_ID' in interactions.columns else interactions.columns[1])
        
        users = interactions[self.user_col].unique()
        items = interactions[self.item_col].unique()
        
        self.user2id = {u: i for i, u in enumerate(users)}
        self.item2id = {it: i for i, it in enumerate(items)}
        self.id2user = {i: u for u, i in self.user2id.items()}
        self.id2item = {i: it for it, i in self.item2id.items()}
        
        self.n_users = len(users)
        self.n_items = len(items)
        self.logger.info(f"Built mappings: {self.n_users} users, {self.n_items} items")
    
    def _build_adjacency_matrix(self, interactions: pd.DataFrame) -> sparse.csr_matrix:
        """Build normalized adjacency matrix for LightGCN."""
        n = self.n_users + self.n_items
        
        rows, cols, data = [], [], []
        for _, row in interactions.iterrows():
            u = row[self.user_col]
            i = row[self.item_col]
            if u not in self.user2id or i not in self.item2id:
                continue
            u_idx = self.user2id[u]
            i_idx = self.n_users + self.item2id[i]
            weight = row.get('INTERACTION', row.get('interaction', row.get('rating', 1.0)))
            rows.extend([u_idx, i_idx])
            cols.extend([i_idx, u_idx])
            data.extend([weight, weight])
        
        R = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        degrees = np.array(R.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
        return (d_inv_sqrt @ R @ d_inv_sqrt).tocsr()
    
    def _compute_embeddings(self, adj_matrix: sparse.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
        """Compute LightGCN embeddings using power iteration."""
        n = self.n_users + self.n_items
        np.random.seed(42)
        embeddings = np.random.normal(0, 0.1, (n, self.embedding_dim))
        
        all_embeddings = [embeddings]
        for _ in range(self.n_layers):
            embeddings = adj_matrix @ embeddings
            all_embeddings.append(embeddings)
        
        final = np.mean(all_embeddings, axis=0)
        user_emb = final[:self.n_users]
        item_emb = final[self.n_users:]
        
        user_emb = user_emb / (np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-8)
        item_emb = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8)
        
        return user_emb, item_emb
    
    def fit(self, interactions: pd.DataFrame) -> 'LightGCNRetriever':
        """Train LightGCN model."""
        self.logger.info("Training LightGCN...")
        self._build_mappings(interactions)
        adj_matrix = self._build_adjacency_matrix(interactions)
        self.logger.info(f"Built adjacency matrix: {adj_matrix.nnz} non-zero entries")
        self.user_embeddings, self.item_embeddings = self._compute_embeddings(adj_matrix)
        self.logger.info(f"Computed embeddings: users {self.user_embeddings.shape}, items {self.item_embeddings.shape}")
        return self
    
    def retrieve(self, user_id: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Retrieve top-K candidates for a user."""
        if user_id not in self.user2id:
            return []
        top_k = top_k or self.top_k
        u_idx = self.user2id[user_id]
        scores = self.item_embeddings @ self.user_embeddings[u_idx]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.id2item[i], float(scores[i])) for i in top_indices]
    
    def batch_retrieve(self, user_ids: List[str], top_k: Optional[int] = None) -> pd.DataFrame:
        """Batch retrieve candidates for multiple users."""
        top_k = top_k or self.top_k
        all_results = []
        for uid in user_ids:
            candidates = self.retrieve(uid, top_k)
            for rank, (iid, score) in enumerate(candidates, 1):
                all_results.append({'user_id': uid, 'item_id': iid, 'retrieval_score': score, 'rank': rank})
        return pd.DataFrame(all_results)
    
    def run(self, interactions: pd.DataFrame) -> Dict[str, Any]:
        """Run the complete retrieval pipeline."""
        self.fit(interactions)
        user_ids = list(self.user2id.keys())
        self.logger.info(f"Retrieving candidates for {len(user_ids)} users...")
        candidates_df = self.batch_retrieve(user_ids, self.top_k)
        return {'n_candidates': len(candidates_df), 'n_users': len(user_ids), 'top_k': self.top_k, 'candidates': candidates_df}
