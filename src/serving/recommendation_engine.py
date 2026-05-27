"""
Recommendation Engine - Real-time serving module for production deployment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json

from ..utils.config_loader import ConfigLoader
from ..utils.helpers import load_json
from .policy_engine import PolicyEngine, PolicyConfig


class RecommendationEngine:
    """
    Production-ready recommendation engine for real-time serving.
    
    Supports:
    - Personalized recommendations
    - Session-based recommendations  
    - Cold-start handling
    - Batch inference
    """
    
    def __init__(self, config: ConfigLoader, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.retriever = None
        self.personalizer = None
        self.ranker = None
        self.reranker = None
        
        # Cache
        self.embedding_cache = {}
        self.recommendation_cache = {}
    
    def load_models(self, checkpoint_dir: str) -> 'RecommendationEngine':
        """Load trained models from checkpoints."""
        self.logger.info(f"Loading models from {checkpoint_dir}")
        # In production, this would load actual model weights
        return self
    
    def get_recommendations(self, user_id: str, 
                            context: Optional[Dict[str, Any]] = None,
                            top_k: int = 20,
                            policy_inputs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a user."""
        policy = self._resolve_policy_inputs(context=context, policy_inputs=policy_inputs)

        # Check cache
        cache_key = f"{user_id}_{top_k}_{hash(policy)}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]
        
        recommendations = []
        
        # Cold start handling
        if not self._has_user_history(user_id):
            recommendations = self._get_cold_start_recommendations(top_k)
        else:
            # Full pipeline
            retrieval_limit = min(policy.candidate_search_limit, top_k * 5)
            retrieved = self._retrieve_candidates(user_id, retrieval_limit, policy)
            personalized = self._personalize(user_id, retrieved, policy)
            ranked = self._rank(personalized, policy)
            reranked = self._rerank(ranked, top_k, policy)
            recommendations = self._inject_promotions(reranked, policy)

        recommendations = self._apply_category_cap(recommendations, policy)
        recommendations = recommendations[:min(top_k, policy.recommendation_yield_limit)]
        
        # Cache results
        self.recommendation_cache[cache_key] = recommendations
        
        return recommendations
    

    def _resolve_policy_inputs(self, context: Optional[Dict[str, Any]], policy_inputs: Optional[Dict[str, Any]]) -> PolicyConfig:
        """Resolve policy inputs from config, context, and direct runtime overrides."""
        resolved: Dict[str, Any] = self.config.get('policy_control', {}) or {}
        if context and isinstance(context.get('policy_control'), dict):
            resolved.update(context['policy_control'])
        if policy_inputs:
            resolved.update(policy_inputs)
        return PolicyEngine.resolve(resolved)

    def _has_user_history(self, user_id: str) -> bool:
        """Check if user has interaction history."""
        # In production, check against stored user history
        return False
    
    def _get_cold_start_recommendations(self, top_k: int) -> List[Dict[str, Any]]:
        """Get popularity-based recommendations for cold-start users."""
        # Return trending/popular items
        return [{'item_id': f'I{i:04d}', 'score': 1.0 - i/top_k, 'reason': 'trending'} 
                for i in range(top_k)]
    
    def _retrieve_candidates(self, user_id: str, n_candidates: int, policy: PolicyConfig) -> List[Dict]:
        """Retrieve candidates using LightGCN."""
        # Placeholder for retrieval
        effective_candidates = min(n_candidates, policy.candidate_search_limit)
        _batch_size = policy.retrieval_batch_size
        return [{'item_id': f'I{i:04d}', 'retrieval_score': np.random.random()} 
                for i in range(effective_candidates)]
    
    def _personalize(self, user_id: str, candidates: List[Dict], policy: PolicyConfig) -> List[Dict]:
        """Apply sequential personalization."""
        for c in candidates:
            c['personalization_score'] = c.get('retrieval_score', 0.5) * np.random.uniform(0.8, 1.2)
            c['freshness_score'] = np.random.random()
            c['freshness_score'] *= policy.recency_decay_coefficient
        return candidates
    
    def _rank(self, candidates: List[Dict], policy: PolicyConfig) -> List[Dict]:
        """Apply DeepFM ranking."""
        for c in candidates:
            existing_rank_score = np.random.random()
            c['ranking_score'] = existing_rank_score * policy.personalization_weight
            exploration_bonus = c.get('novelty_score', 0.0) + c.get('long_tail_score', 0.0) + c.get('uncertainty_score', 0.0)
            c['ranking_score'] += policy.exploration_weight * exploration_bonus
        return sorted(candidates, key=lambda x: x['ranking_score'], reverse=True)
    
    def _rerank(self, candidates: List[Dict], top_k: int, policy: PolicyConfig) -> List[Dict]:
        """Apply diversity re-ranking."""
        result = []
        categories_seen = {}
        
        for c in candidates[:top_k * 2]:
            cat = 'default'  # Would get from item features
            dynamic_category_cap = max(1, int(round((1.0 - policy.candidate_output_diversity_index) * policy.category_capping_threshold)))
            if categories_seen.get(cat, 0) < dynamic_category_cap:
                c['final_rank'] = len(result) + 1
                result.append(c)
                categories_seen[cat] = categories_seen.get(cat, 0) + 1
            
            if len(result) >= top_k:
                break
        
        return result
    

    def _inject_promotions(self, candidates: List[Dict], policy: PolicyConfig) -> List[Dict]:
        """Inject sponsored promotions after reranking when threshold criteria is met."""
        if not candidates or policy.promotion_weight <= 0:
            return candidates

        result = list(candidates)
        threshold_index = int(len(result) * policy.promotions_injection_percentile_threshold)
        threshold_index = min(max(0, threshold_index), len(result) - 1)
        if threshold_index < len(result):
            result[-1] = {**result[-1], 'is_sponsored': True, 'promotion_weight': policy.promotion_weight}
        return result

    def _apply_category_cap(self, candidates: List[Dict], policy: PolicyConfig) -> List[Dict]:
        """Apply category cap only in final filtering stage."""
        filtered = []
        categories_seen = {}
        for candidate in candidates:
            cat = candidate.get('category', 'default')
            if categories_seen.get(cat, 0) >= policy.category_capping_threshold:
                continue
            filtered.append(candidate)
            categories_seen[cat] = categories_seen.get(cat, 0) + 1
        return filtered

    def batch_recommend(self, user_ids: List[str], top_k: int = 20) -> pd.DataFrame:
        """Batch generate recommendations for multiple users."""
        all_recs = []
        for user_id in user_ids:
            recs = self.get_recommendations(user_id, top_k=top_k)
            for rec in recs:
                rec['user_id'] = user_id
                all_recs.append(rec)
        return pd.DataFrame(all_recs)
    
    def explain_recommendation(self, user_id: str, item_id: str) -> Dict[str, Any]:
        """Generate explanation for a recommendation."""
        return {
            'user_id': user_id,
            'item_id': item_id,
            'explanation': {
                'retrieval_reason': 'Based on your browsing history',
                'sequential_reason': 'Users who viewed similar items also liked this',
                'feature_contributions': [],
                'diversity_adjustment': 'Added for category diversity'
            }
        }
