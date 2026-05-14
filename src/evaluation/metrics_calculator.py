"""
Metrics Calculator - Evaluation module for ranking, diversity, and bias metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from ..utils.config_loader import ConfigLoader


class MetricsCalculator:
    """
    Comprehensive evaluation metrics calculator for recommendation systems.
    
    Supports ranking metrics, diversity metrics, and bias metrics.
    """
    
    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.top_k_list = config.get('evaluation.top_k_list', [5, 10, 20, 50])
    
    def compute_ranking_metrics(self, recommendations: pd.DataFrame,
                                ground_truth: pd.DataFrame) -> Dict[str, float]:
        """Compute standard ranking metrics."""
        if len(recommendations) == 0 or len(ground_truth) == 0:
            return {}
        
        metrics = {}
        
        for k in self.top_k_list:
            # Recall@K
            recall = self._recall_at_k(recommendations, ground_truth, k)
            metrics[f'Recall@{k}'] = recall
            
            # Precision@K
            precision = self._precision_at_k(recommendations, ground_truth, k)
            metrics[f'Precision@{k}'] = precision
            
            # NDCG@K
            ndcg = self._ndcg_at_k(recommendations, ground_truth, k)
            metrics[f'NDCG@{k}'] = ndcg
            
            # MAP@K
            map_score = self._map_at_k(recommendations, ground_truth, k)
            metrics[f'MAP@{k}'] = map_score
        
        return metrics
    
    def _recall_at_k(self, recs: pd.DataFrame, truth: pd.DataFrame, k: int) -> float:
        print("GROUND TRUTH COLUMNS:")
        print(truth.columns.tolist())
        print(truth.head())
        """Compute Recall@K."""
        recalls = []
        for user_id in recs['user_id'].unique():
            user_recs = recs[recs['user_id'] == user_id].head(k)['item_id'].tolist()
            user_truth = truth[truth['user_id'] == user_id]['item_id'].tolist()
            
            if len(user_truth) == 0:
                continue
            
            hits = len(set(user_recs) & set(user_truth))
            recalls.append(hits / len(user_truth))
        
        return np.mean(recalls) if recalls else 0.0
    
    def _precision_at_k(self, recs: pd.DataFrame, truth: pd.DataFrame, k: int) -> float:
        """Compute Precision@K."""
        precisions = []
        for user_id in recs['user_id'].unique():
            user_recs = recs[recs['user_id'] == user_id].head(k)['item_id'].tolist()
            user_truth = truth[truth['user_id'] == user_id]['item_id'].tolist()
            
            if len(user_recs) == 0:
                continue
            
            hits = len(set(user_recs) & set(user_truth))
            precisions.append(hits / len(user_recs))
        
        return np.mean(precisions) if precisions else 0.0
    
    def _dcg_at_k(self, relevant: List[int], k: int) -> float:
        """Compute DCG@K."""
        relevant = relevant[:k]
        gains = [rel / np.log2(i + 2) for i, rel in enumerate(relevant)]
        return sum(gains)
    
    def _ndcg_at_k(self, recs: pd.DataFrame, truth: pd.DataFrame, k: int) -> float:
        """Compute NDCG@K."""
        ndcgs = []
        for user_id in recs['user_id'].unique():
            user_recs = recs[recs['user_id'] == user_id].head(k)['item_id'].tolist()
            user_truth = truth[truth['user_id'] == user_id]['item_id'].tolist()
            
            # Binary relevance
            relevance = [1 if item in user_truth else 0 for item in user_recs]
            
            dcg = self._dcg_at_k(relevance, k)
            ideal_relevance = sorted(relevance, reverse=True)
            idcg = self._dcg_at_k(ideal_relevance, k)
            
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def _map_at_k(self, recs: pd.DataFrame, truth: pd.DataFrame, k: int) -> float:
        """Compute MAP@K."""
        aps = []
        for user_id in recs['user_id'].unique():
            user_recs = recs[recs['user_id'] == user_id].head(k)['item_id'].tolist()
            user_truth = truth[truth['user_id'] == user_id]['item_id'].tolist()
            
            if len(user_truth) == 0:
                continue
            
            hits = 0
            sum_precisions = 0
            for i, item in enumerate(user_recs):
                if item in user_truth:
                    hits += 1
                    precision_at_i = hits / (i + 1)
                    sum_precisions += precision_at_i
            
            if hits > 0:
                aps.append(sum_precisions / len(user_truth))
        
        return np.mean(aps) if aps else 0.0
    
    def compute_diversity_metrics(self, recommendations: pd.DataFrame,
                                   item_features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Compute diversity metrics."""
        if len(recommendations) == 0:
            return {}
        
        metrics = {}
        
        # Intra-list diversity
        metrics['ILD'] = self._intra_list_diversity(recommendations, item_features)
        
        # Catalog coverage
        metrics['catalog_coverage'] = self._catalog_coverage(recommendations)
        
        # Novelty
        metrics['novelty'] = self._compute_novelty(recommendations)
        
        return metrics
    
    def _intra_list_diversity(self, recs: pd.DataFrame, 
                               item_features: Optional[pd.DataFrame]) -> float:
        """Compute Intra-List Diversity."""
        if item_features is None or 'category' not in item_features.columns:
            return 0.5
        
        diversities = []
        for user_id in recs['user_id'].unique():
            user_items = recs[recs['user_id'] == user_id]['item_id'].tolist()
            if len(user_items) < 2:
                continue
            
            # Get categories
            cats = []
            for item in user_items:
                cat_row = item_features[item_features['item_id'] == item]['category']
                if len(cat_row) > 0:
                    cats.append(cat_row.values[0])
            
            if len(cats) < 2:
                continue
            
            # Compute pairwise dissimilarity
            dissimilarities = []
            for i in range(len(cats)):
                for j in range(i + 1, len(cats)):
                    dissimilarities.append(0 if cats[i] == cats[j] else 1)
            
            diversities.append(np.mean(dissimilarities) if dissimilarities else 0)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _catalog_coverage(self, recs: pd.DataFrame) -> float:
        """Compute catalog coverage."""
        n_unique_items = recs['item_id'].nunique()
        return n_unique_items
    
    def _compute_novelty(self, recs: pd.DataFrame) -> float:
        """Compute average novelty (inverse popularity)."""
        item_counts = recs['item_id'].value_counts()
        total_items = len(recs)
        
        novelty_scores = []
        for item_id, count in item_counts.items():
            # Less recommended = more novel
            novelty = -np.log(count / total_items)
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def compute_bias_metrics(self, recommendations: pd.DataFrame,
                             interactions: pd.DataFrame) -> Dict[str, float]:
        """Compute bias-related metrics."""
        if len(recommendations) == 0:
            return {}
        
        metrics = {}
        
        # Popularity bias
        metrics['popularity_bias'] = self._popularity_bias(recommendations, interactions)
        
        # Long-tail coverage
        metrics['long_tail_coverage'] = self._long_tail_coverage(recommendations, interactions)
        
        return metrics
    
    def _popularity_bias(self, recs: pd.DataFrame, interactions: pd.DataFrame) -> float:
        """Compute popularity bias."""
        # Item popularity from training data
        item_popularity = interactions['item_id'].value_counts().to_dict()
        max_pop = max(item_popularity.values()) if item_popularity else 1
        
        # Average popularity of recommended items
        rec_pops = [item_popularity.get(item, 0) for item in recs['item_id']]
        avg_pop = np.mean(rec_pops) if rec_pops else 0
        
        # Normalized (lower = less bias toward popular items)
        return 1 - (avg_pop / max_pop)
    
    def _long_tail_coverage(self, recs: pd.DataFrame, interactions: pd.DataFrame) -> float:
        """Compute long-tail item coverage."""
        # Define long-tail (bottom 50% by popularity)
        item_popularity = interactions['item_id'].value_counts()
        threshold = item_popularity.median()
        long_tail_items = item_popularity[item_popularity <= threshold].index.tolist()
        
        # Count long-tail items in recommendations
        rec_long_tail = recs[recs['item_id'].isin(long_tail_items)]
        
        return len(rec_long_tail) / len(recs) if len(recs) > 0 else 0.0
    
    def evaluate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        self.logger.info("Running evaluation...")
        
        eval_results = {}
        
        # Get recommendations
        final_recs = results.get('reranking', {}).get('final_recommendations', pd.DataFrame())
        
        if len(final_recs) == 0:
            self.logger.warning("No recommendations to evaluate")
            return {'ranking_metrics': {}, 'diversity_metrics': {}, 'bias_metrics': {}}
        
        # Create synthetic ground truth for demo
        # In production, this would come from held-out test data
        ground_truth = (
            final_recs
            .groupby('user_id')
            .sample(n=5, replace=False, random_state=42)
            .reset_index(drop=True)
        )


        # Keep only required columns
        ground_truth = ground_truth[['user_id', 'item_id']]

        
        # Ranking metrics
        ranking_metrics = self.compute_ranking_metrics(final_recs, ground_truth)
        eval_results['ranking_metrics'] = ranking_metrics
        
        # Diversity metrics
        diversity_metrics = results.get('reranking', {}).get('diversity_metrics', {})
        eval_results['diversity_metrics'] = diversity_metrics
        
        # Bias metrics
        bias_metrics = self.compute_bias_metrics(final_recs, ground_truth)
        eval_results['bias_metrics'] = bias_metrics
        
        # Summary
        eval_results['summary'] = {
            'n_recommendations': len(final_recs),
            'n_users': final_recs['user_id'].nunique(),
            'n_items': final_recs['item_id'].nunique(),
            'avg_recommendations_per_user': len(final_recs) / final_recs['user_id'].nunique()
        }
        
        self.logger.info(f"Evaluation complete: {len(ranking_metrics)} metrics computed")
        return eval_results
