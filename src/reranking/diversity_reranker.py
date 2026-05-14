"""
Diversity Reranker - Stage 4: Diversity-aware re-ranking and bias mitigation.
Implements MMR and other diversity optimization techniques.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from ..utils.config_loader import ConfigLoader


class DiversityReranker:
    """
    Diversity-aware re-ranker for final recommendation optimization.
    
    Implements Maximum Marginal Relevance (MMR) and other 
    diversity/bias mitigation techniques.
    """
    
    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # MMR parameters
        self.lambda_param = config.get('reranking.mmr.lambda_param', 0.7)
        
        # Diversity weights
        self.diversity_weight = config.get('reranking.weights.diversity', 0.3)
        self.novelty_weight = config.get('reranking.weights.novelty', 0.2)
        self.freshness_weight = config.get('reranking.weights.freshness', 0.1)
        
        # Balancing constraints
        self.max_same_category = config.get('reranking.category_balance.max_same_category', 5)
        self.min_categories = config.get('reranking.category_balance.min_categories', 3)
        self.max_same_seller = config.get('reranking.seller_balance.max_same_seller', 3)
        
        # Final output
        self.final_top_k = config.get('reranking.final_top_k', 20)
    
    def _compute_diversity_score(self, item_id: str, selected_items: List[str], 
                                  item_features: Optional[pd.DataFrame] = None) -> float:
        """Compute diversity score based on dissimilarity to selected items."""
        if not selected_items or item_features is None:
            return 0.5
        
        # Simple category-based diversity
        if 'category' in item_features.columns and 'item_id' in item_features.columns:
            item_cat = item_features[item_features['item_id'] == item_id]['category'].values
            if len(item_cat) == 0:
                return 0.5
            
            selected_cats = []
            for sel_item in selected_items[-self.max_same_category:]:
                sel_cat = item_features[item_features['item_id'] == sel_item]['category'].values
                if len(sel_cat) > 0:
                    selected_cats.append(sel_cat[0])
            
            # Higher score if category is different from recent selections
            if item_cat[0] not in selected_cats:
                return 1.0
            else:
                return 0.3
        
        return 0.5
    
    def _compute_novelty_score(self, item_id: str, user_history: List[str],
                                item_popularity: Optional[Dict[str, int]] = None) -> float:
        """Compute novelty score (inverse popularity)."""
        if item_popularity is None:
            return 0.5
        
        max_pop = max(item_popularity.values()) if item_popularity else 1
        item_pop = item_popularity.get(item_id, max_pop // 2)
        
        # Novelty is inverse of popularity
        novelty = 1.0 - (item_pop / (max_pop + 1))
        return novelty
    
    def _compute_freshness_score(self, item_id: str, 
                                   item_features: Optional[pd.DataFrame] = None) -> float:
        """Compute freshness score based on item recency."""
        if item_features is None or 'created_date' not in item_features.columns:
            return 0.5
        
        from datetime import datetime
        item_row = item_features[item_features['item_id'] == item_id]
        if len(item_row) == 0:
            return 0.5
        
        created = item_row['created_date'].values[0]
        if isinstance(created, np.datetime64):
            created = pd.Timestamp(created)
        
        age_days = (datetime.now() - created).days
        # Decay over 30 days
        freshness = np.exp(-age_days / 30)
        return freshness
    
    def mmr_rerank(self, candidates: pd.DataFrame, 
                   item_features: Optional[pd.DataFrame] = None,
                   user_history: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Apply Maximum Marginal Relevance re-ranking.
        
        MMR = lambda * relevance - (1-lambda) * diversity
        """
        if len(candidates) == 0:
            return candidates
        
        reranked = []
        remaining = candidates.copy()
        
        # Get item popularity for novelty calculation
        item_popularity = {}
        if 'item_id' in candidates.columns:
            item_counts = candidates['item_id'].value_counts().to_dict()
            item_popularity = item_counts
        
        while len(remaining) > 0 and len(reranked) < self.final_top_k:
            best_score = -np.inf
            best_idx = 0
            
            selected_ids = [r['item_id'] for r in reranked]
            
            for idx, row in remaining.iterrows():
                item_id = row['item_id']
                
                # Get base relevance score
                relevance = row.get('ranking_score', row.get('personalization_score', 
                               row.get('retrieval_score', 0.5)))
                
                # Compute diversity components
                diversity = self._compute_diversity_score(item_id, selected_ids, item_features)
                novelty = self._compute_novelty_score(item_id, [], item_popularity)
                freshness = self._compute_freshness_score(item_id, item_features)
                
                # MMR score
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * (
                        self.diversity_weight * diversity +
                        self.novelty_weight * novelty +
                        self.freshness_weight * freshness
                    )
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            # Add best item to reranked list
            best_row = remaining.loc[best_idx].copy()
            best_row['mmr_score'] = best_score
            best_row['diversity_score'] = self._compute_diversity_score(
                best_row['item_id'], selected_ids, item_features
            )
            reranked.append(best_row)
            remaining = remaining.drop(best_idx)
        
        if len(reranked) == 0:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(reranked)
        result_df['final_rank'] = range(1, len(result_df) + 1)
        return result_df
    
    def apply_constraints(self, recommendations: pd.DataFrame,
                          item_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply business constraints to recommendations."""
        if len(recommendations) == 0:
            return recommendations
        
        filtered = []
        category_counts = {}
        seller_counts = {}
        
        for _, row in recommendations.iterrows():
            item_id = row['item_id']
            
            # Get item attributes
            category = None
            seller = None
            if item_features is not None:
                item_row = item_features[item_features['item_id'] == item_id]
                if len(item_row) > 0:
                    category = item_row['category'].values[0] if 'category' in item_row.columns else None
                    seller = item_row['brand'].values[0] if 'brand' in item_row.columns else None
            
            # Check category constraint
            if category:
                current_count = category_counts.get(category, 0)
                if current_count >= self.max_same_category:
                    continue
                category_counts[category] = current_count + 1
            
            # Check seller constraint
            if seller:
                current_count = seller_counts.get(seller, 0)
                if current_count >= self.max_same_seller:
                    continue
                seller_counts[seller] = current_count + 1
            
            filtered.append(row.to_dict())
        
        result = pd.DataFrame.from_records(filtered)
        if len(result) > 0:
            result['final_rank'] = range(1, len(result) + 1)
        
        return result
    
    def run(self, ranking_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete re-ranking pipeline."""
        self.logger.info("Running diversity re-ranking...")
        
        ranked_candidates = ranking_results.get('ranked_candidates', pd.DataFrame())
        
        if len(ranked_candidates) == 0:
            return {
                'n_final': 0,
                'final_recommendations': pd.DataFrame(),
                'diversity_metrics': {}
            }
        
        # Group by user and apply MMR
        all_recommendations = []
        
        for user_id, group in ranked_candidates.groupby('user_id'):
            user_recs = self.mmr_rerank(group.sort_values('ranking_score', ascending=False).head(50))
            user_recs['user_id'] = user_id
            all_recommendations.append(user_recs)
        
        if len(all_recommendations) == 0:
            return {'n_final': 0, 'final_recommendations': pd.DataFrame()}
        
        final_df = pd.concat(all_recommendations, ignore_index=True)
        
        # Apply constraints
        final_df = self.apply_constraints(final_df)
        self.logger.info(f"Final columns: {final_df.columns.tolist()}")
        
        # Compute diversity metrics
        diversity_metrics = self._compute_diversity_metrics(final_df)
        
        results = {
            'n_final': len(final_df),
            'n_users': final_df['user_id'].nunique() if 'user_id' in final_df.columns else 0,
            'final_top_k': self.final_top_k,
            'final_recommendations': final_df,
            'diversity_metrics': diversity_metrics
        }
        
        self.logger.info(f"Generated {len(final_df)} final recommendations with diversity metrics")
        return results
    
    def _compute_diversity_metrics(self, recommendations: pd.DataFrame) -> Dict[str, float]:
        """Compute diversity metrics for recommendations."""
        if len(recommendations) == 0:
            return {}
        
        metrics = {}
        
        # Intra-list diversity (ILD)
        if 'diversity_score' in recommendations.columns:
            metrics['avg_diversity_score'] = float(recommendations['diversity_score'].mean())
        
        # Category coverage
        metrics['n_unique_items'] = recommendations['item_id'].nunique()
        
        # User coverage
        if 'user_id' in recommendations.columns:
            metrics['n_users_covered'] = recommendations['user_id'].nunique()
        
        return metrics
