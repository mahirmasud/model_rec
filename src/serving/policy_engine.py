"""Policy control wrapper for safe, bounded runtime recommendation controls."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


Number = Union[int, float]


@dataclass(frozen=True)
class PolicyConfig:
    """Normalized policy controls consumed by existing pipeline stages."""

    recommendation_yield_limit: int
    personalization_weight: float
    exploration_weight: float
    promotion_weight: float
    candidate_output_diversity_index: float
    retrieval_batch_size: int
    candidate_search_limit: int
    recency_decay_coefficient: float
    category_capping_threshold: int
    promotions_injection_percentile_threshold: float


class PolicyEngine:
    """Clamp and normalize user policy inputs into a safe configuration object."""

    DEFAULTS: Dict[str, Any] = {
        "recommendation_yield_limit": 20,
        "vector_personalization_focus": 0.6,
        "curation_discovery_factor": 0.25,
        "corporate_sponsored_promotions": 0.0,
        "candidate_output_diversity_index": 0.2,
        "retrieval_batch_size": 256,
        "candidate_search_limit": 300,
        "recency_decay_coefficient": 0.8,
        "category_capping_threshold": 3,
        "promotions_injection_percentile_threshold": 0.9,
    }

    @staticmethod
    def _clamp(value: Number, min_value: Number, max_value: Number) -> Number:
        return max(min_value, min(max_value, value))

    @classmethod
    def _to_weight(cls, value: Any) -> float:
        numeric = float(value)
        # Accept percentage-style values and normalize.
        if numeric > 1.0:
            numeric = numeric / 100.0
        return float(cls._clamp(numeric, 0.0, 1.0))

    @classmethod
    def _promotion_weight(cls, value: Any) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return cls._to_weight(value)

    @classmethod
    def resolve(cls, user_inputs: Optional[Dict[str, Any]] = None) -> PolicyConfig:
        raw = dict(cls.DEFAULTS)
        raw.update(user_inputs or {})

        personalization_weight = cls._to_weight(raw["vector_personalization_focus"])
        exploration_weight = cls._to_weight(raw["curation_discovery_factor"])
        promotion_weight = cls._promotion_weight(raw["corporate_sponsored_promotions"])

        total = personalization_weight + exploration_weight + promotion_weight
        if total > 1.0:
            scale = 1.0 / total
            personalization_weight *= scale
            exploration_weight *= scale
            promotion_weight *= scale

        return PolicyConfig(
            recommendation_yield_limit=int(
                cls._clamp(int(raw["recommendation_yield_limit"]), 1, 50)
            ),
            personalization_weight=personalization_weight,
            exploration_weight=exploration_weight,
            promotion_weight=promotion_weight,
            candidate_output_diversity_index=float(
                cls._clamp(float(raw["candidate_output_diversity_index"]), 0.0, 1.0)
            ),
            retrieval_batch_size=int(cls._clamp(int(raw["retrieval_batch_size"]), 64, 512)),
            candidate_search_limit=int(cls._clamp(int(raw["candidate_search_limit"]), 100, 500)),
            recency_decay_coefficient=float(
                cls._clamp(float(raw["recency_decay_coefficient"]), 0.1, 0.96)
            ),
            category_capping_threshold=int(
                cls._clamp(int(raw["category_capping_threshold"]), 1, 10)
            ),
            promotions_injection_percentile_threshold=float(
                cls._clamp(float(raw["promotions_injection_percentile_threshold"]), 0.0, 1.0)
            ),
        )
